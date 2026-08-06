"""Post-solve analysis for the maritime chokepoint model.

Result tables, publication figures, correctness invariants, and robustness sweeps. Depends only on model.
"""
from __future__ import annotations

import sys
import model
from model import *          # optimization stack (public names)
from model import _detect_solver

# Aliases kept for the former split-package call sites.
scen = model                 # validation.py: `from . import model as scen`
mt = sys.modules[__name__]        # results/figures alias `from . import results as mt`


# RESULTS: tables, attribution, diagnostics built from the solved MIP.

import gc
import time
from itertools import combinations, chain
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo



# Emergency substitute-supply capability as a share of demand (P6 total, ~5%); credited in the ESCVI alternative-supply gap.
SUBSTITUTE_ALT_FRAC: float = 0.05

REACTIVE: Tuple[str, ...] = ("P1", "P2", "P3")

# Cache solved models by solve args so country/regional/sector tables reuse one solve.
_solve_cache: Dict[tuple, "pyo.ConcreteModel"] = {}


def solve(policy_subset: Sequence[str] = ALL_POLICIES,
          scenario_class: str = "full_closure",
          chokepoint: str = "CP_HORMUZ",
          duration_days: int = 100,
          stage_days: Optional[int] = None,
          enable_cascade: bool = True,
          service_floor: float = 0.60,
          equity_weight: float = 1.0,
          target_count: int = 15,
          cache: bool = True) -> "pyo.ConcreteModel":
    """Build and solve the MIP; return the solved model.

    Only reused Hormuz solves are cached; sweeps pass cache=False to release each large model after reading its scalars.
    """
    disrupted = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    choke_label = "+".join(disrupted) if isinstance(disrupted, list) else disrupted
    key = (tuple(sorted(policy_subset)), scenario_class, choke_label,
           duration_days, stage_days, enable_cascade, round(service_floor, 4),
           round(equity_weight, 4), target_count)
    if cache and key in _solve_cache:
        return _solve_cache[key]
    net = get_network()
    scen = get_scenarios(scenario_class, target_count)
    m = build_model(net, scen, disrupted_chokepoint=disrupted,
                    duration_days=duration_days, enable_cascade=enable_cascade,
                    unresponsive_regime=class_regime(scenario_class)[1],
                    policy_subset=list(policy_subset), stage_days=stage_days,
                    service_floor=service_floor, equity_weight=equity_weight)
    info = solve_model(m, solver_name=_detect_solver(None),
                       time_limit=600, mip_gap=1e-3)
    if info["objective"] is None:
        raise RuntimeError(
            f"Solver returned no solution (status: {info['status']}) for "
            f"subset={policy_subset}, chokepoint={chokepoint}, class={scenario_class}. "
            f"This model is feasible and solves in isolation, so a no-incumbent "
            f"return almost always means the solver ran out of memory. Free RAM "
            f"(close other applications) and rerun.")
    m._objective_value = float(info["objective"])
    if cache:
        _solve_cache[key] = m
        # Keep only the 4 most recent solves (they are large).
        while len(_solve_cache) > 4:
            _solve_cache.pop(next(iter(_solve_cache)))
    return m


# Extraction primitives
def _cscale(m, n, s, t) -> float:
    """Economic-size scaling of the amplification term (1.0 if the model predates the scaled formulation)."""
    return pyo.value(m.casc_scale[n, s, t]) if hasattr(m, "casc_scale") else 1.0


def country_damage(m, include_price: bool = True) -> Dict[str, float]:
    """Per-country expected economic loss (USD billion) from the solved model.

    Channels: direct shortage (pi*h), cross-sector cascade (eta*Delta), and a share of the global oil-price welfare cost. include_price=False gives shortage + cascade only.
    """
    has_casc = hasattr(m, "Delta")
    pw = realized_scenario_probs(m)
    use_price = include_price and hasattr(m, "dwl")

    # Attribute the single world-price deadweight term by served-consumption share (unserved barrels already charged by the shortage term), so shares telescope to the aggregate.
    price_total = 0.0
    qshare = {n: 0.0 for n in m.C}
    if use_price:
        for t in m.T:
            for w in m.OMEGA:
                price_total += pw[w] * pyo.value(m.dwl[t, w])
        for n in m.C:
            for s in m.S:
                for t in m.T:
                    for w in m.OMEGA:
                        qshare[n] += pw[w] * pyo.value(m.q[n, s, t, w])
        q_tot = sum(qshare.values()) or 1.0

    out: Dict[str, float] = {}
    for n in m.C:
        tot = 0.0
        for s in m.S:
            pi = pyo.value(m.pi_ns[n, s])
            eta = pyo.value(m.eta_ns[n, s]) if has_casc else 0.0
            for t in m.T:
                for w in m.OMEGA:
                    tot += pw[w] * pi * pyo.value(m.h[n, s, t, w])
                    if has_casc:
                        tot += pw[w] * eta * _cscale(m, n, s, t) * pyo.value(m.Delta[n, s, t, w])
        if use_price:
            tot += price_total * qshare[n] / q_tot
        out[n] = tot / 1000.0
    return out


def country_access(m) -> Dict[str, float]:
    """Per-country oil-access ratio (prob-weighted served / demand) in [0,1]."""
    pw = realized_scenario_probs(m)
    out: Dict[str, float] = {}
    for n in m.C:
        served = dem = 0.0
        for s in m.S:
            for t in m.T:
                d = pyo.value(m.demand[n, s, t])
                if d <= 0:
                    continue
                for w in m.OMEGA:
                    served += pw[w] * pyo.value(m.q[n, s, t, w])
                    dem += pw[w] * d
        out[n] = (served / dem) if dem > 0 else 1.0
    return out


def sector_damage(m, code: str) -> Dict[str, Dict[str, float]]:
    """Per-sector direct-shortage and cascade damage (USD billion) for one country."""
    has_casc = hasattr(m, "Delta")
    pw = realized_scenario_probs(m)
    out: Dict[str, Dict[str, float]] = {}
    for s in m.S:
        pi = pyo.value(m.pi_ns[code, s])
        eta = pyo.value(m.eta_ns[code, s]) if has_casc else 0.0
        direct = casc = 0.0
        for t in m.T:
            for w in m.OMEGA:
                direct += pw[w] * pi * pyo.value(m.h[code, s, t, w]) / 1000.0
                if has_casc:
                    casc += pw[w] * eta * _cscale(m, code, s, t) * pyo.value(m.Delta[code, s, t, w]) / 1000.0
        out[s] = {"direct": direct, "cascade": casc, "total": direct + casc}
    return out


def cost_components(m) -> Dict[str, float]:
    """Objective decomposition (USD billion): transport (with maritime freight/war-risk premium), policy activation, direct shortage, cascade, oil-price increase, and priority-floor equity penalty.
    """
    pw = realized_scenario_probs(m)
    has = hasattr(m, "Delta")
    mar = getattr(m, "_maritime_ids", set())
    transport = sum(pw[w] * (pyo.value(m.mar_mult[t, w]) if a in mar else 1.0)
                    * pyo.value(m.arc_cost[a]) * pyo.value(m.x[a, t, w])
                    for a in m.A for t in m.T for w in m.OMEGA) / 1e3
    policy = sum(pw[w] * pyo.value(m.gamma[p]) * pyo.value(m.z[p, t, w])
                 for p in m.P for t in m.T for w in m.OMEGA) / 1e3
    direct = sum(pw[w] * pyo.value(m.equity_mult[n, s]) * pyo.value(m.pi_ns[n, s])
                 * pyo.value(m.h[n, s, t, w])
                 for n in m.C for s in m.S for t in m.T for w in m.OMEGA) / 1e3
    cascade = (sum(pw[w] * pyo.value(m.equity_mult[n, s]) * pyo.value(m.eta_ns[n, s])
                   * _cscale(m, n, s, t) * pyo.value(m.Delta[n, s, t, w])
                   for n in m.C for s in m.S for t in m.T for w in m.OMEGA) / 1e3) if has else 0.0
    # Price channel enters welfare as the deadweight loss only (real resource cost of the price rise), not the terms-of-trade transfer.
    price = (sum(pw[w] * pyo.value(m.dwl[t, w])
                 for t in m.T for w in m.OMEGA) / 1e3) \
        if hasattr(m, "dwl") else 0.0
    floor = (sum(pw[w] * FLOOR_PENALTY * pyo.value(m.fslack[n, s, t, w])
                 for (n, s) in m.K_PRIO for t in m.T for w in m.OMEGA) / 1e3) \
        if hasattr(m, "fslack") else 0.0
    curtail = sum(
        pw[w] * (EFFICIENCY_COST_FRAC * BASELINE_OIL_PRICE * EFFICIENCY_FRAC
                 + pyo.value(m.equity_mult[n, s]) * pyo.value(m.pi_ns[n, s])
                 * (1.0 - EFFICIENCY_FRAC))
        * pyo.value(m.rho[n, s, t, w]) * pyo.value(m.demand[n, s, t])
        for n in m.C for s in m.S for t in m.T for w in m.OMEGA) / 1e3
    reserve = (sum(pw[w] * RESERVE_REPLACEMENT_COST * pyo.value(m.r[n, t, w])
                   for n in m.C for t in m.T for w in m.OMEGA) / 1e3) \
        if hasattr(m, "r") else 0.0
    substitute = (sum(pw[w] * (SUBSTITUTE_COST_PS * pyo.value(m.g_ps[n, t, w])
                               + SUBSTITUTE_COST_RF * pyo.value(m.g_rf[n, t, w])
                               + SUBSTITUTE_COST_FS * pyo.value(m.g_fs[n, t, w]))
                      for n in m.C for t in m.T for w in m.OMEGA) / 1e3) \
        if hasattr(m, "g_ps") else \
        ((sum(pw[w] * SUBSTITUTE_SUPPLY_COST * pyo.value(m.g[n, t, w])
              for n in m.C for t in m.T for w in m.OMEGA) / 1e3)
         if hasattr(m, "g") else 0.0)
    out = {"Transport": transport, "Policy": policy, "Direct": direct,
           "Cascade": cascade, "Oil price": price, "Equity floor": floor,
           "Curtailment": curtail, "Reserve": reserve, "Substitute": substitute}
    out["Total"] = sum(v for k, v in out.items() if k != "Total")
    # Memos (excluded from welfare Total). Terms-of-trade transfer to producers is a redistribution, levied on cleared consumption Q0(1 - dQ/Qbar); importer expenditure adds it to the welfare loss.
    if hasattr(m, "price_loss"):
        ws = float(WORLD_OIL_SUPPLY_MBD * STAGE_DAYS)
        transfer = sum(pw[w] * pyo.value(m.price_coef[t]) * pyo.value(m.price_loss[t, w])
                       * max(0.0, 1.0 - pyo.value(m.price_loss[t, w]) / ws)
                       for t in m.T for w in m.OMEGA) / 1e3
        out["Transfer (memo)"] = transfer
        out["Importer expenditure (memo)"] = out["Total"] + transfer
    return out


def gini(values: np.ndarray) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * v)) / (n * np.sum(v)) - (n + 1) / n)


# Result tables (all MIP-derived)
def top_absolute_losses(top_n: int = 10, target_count: int = 15) -> pd.DataFrame:
    """Per-country inaction vs. proactive expected loss, ranked (MIP)."""
    net = get_network()
    m0 = solve((), target_count=target_count, cache=False)
    d0 = country_damage(m0)
    codes = list(m0.C)
    del m0
    gc.collect()
    mp = solve(ALL_POLICIES, target_count=target_count, cache=False)
    dp = country_damage(mp)
    del mp
    gc.collect()
    total0 = sum(d0.values())
    rows = []
    for n in codes:
        c = net.countries[n]
        rows.append({
            "Country": c.name, "Code": n, "Region": c.region,
            "Inaction loss ($B)": round(d0[n], 1),
            "Early-stage loss ($B)": round(dp[n], 1),
            "Saving ($B)": round(d0[n] - dp[n], 1),
            "Global share (%)": round(100.0 * d0[n] / total0, 1) if total0 else 0.0,
        })
    df = pd.DataFrame(rows).sort_values("Inaction loss ($B)", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def regional_losses(target_count: int = 15) -> pd.DataFrame:
    """Regional expected loss under inaction / reactive / proactive (MIP)."""
    net = get_network()
    dmg = {}
    codes = []
    for label, subset in [("No intervention", ()), ("Reactive only", REACTIVE),
                          ("Proactive", ALL_POLICIES)]:
        m = solve(subset, target_count=target_count, cache=False)
        dmg[label] = country_damage(m)
        codes = list(m.C)
        del m
        gc.collect()
    regions = sorted({net.countries[n].region for n in codes})
    rows = []
    for region in regions:
        rc = [n for n in codes if net.countries[n].region == region]
        row = {"Region": region, "Countries": len(rc)}
        for regime, d in dmg.items():
            row[regime + " ($B)"] = round(sum(d[n] for n in rc), 1)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("No intervention ($B)", ascending=False)


def transferability(target_count: int = 15) -> pd.DataFrame:
    """Inaction vs. proactive total expected loss for Hormuz, Malacca, Suez (MIP).

    Loss and saving use the full MIP objective (system-level cost-benefit view), unlike the per-country and regional tables which report only shortage + cascade borne by each economy.
    """
    cps = [("Strait of Hormuz", "CP_HORMUZ", 20.5),
           ("Strait of Malacca", "CP_MALACCA", 16.0),
           ("Suez Canal", "CP_SUEZ", 5.5)]
    rows = []
    for name, cid, tput in cps:
        m0 = solve((), chokepoint=cid, target_count=target_count, cache=False)
        l0 = m0._objective_value / 1000.0
        del m0
        gc.collect()
        mp = solve(ALL_POLICIES, chokepoint=cid, target_count=target_count, cache=False)
        lp = mp._objective_value / 1000.0
        del mp
        gc.collect()
        rows.append({
            "Disrupted chokepoint": name, "Throughput (mbd)": tput,
            "Inaction loss ($B)": round(l0, 1),
            "Proactive loss ($B)": round(lp, 1),
            "Total saving (%)": f"{(1 - lp / l0) * 100:.1f}%" if l0 else "n/a",
        })
    return pd.DataFrame(rows)


def sector_decomposition(code: str = "PAK", target_count: int = 15) -> pd.DataFrame:
    """Per-sector inaction damage decomposition (direct vs cascade) for a country."""
    m0 = solve((), target_count=target_count, cache=False)
    sd = sector_damage(m0, code)
    del m0
    gc.collect()
    rows = []
    for s in SECTORS:
        d = sd[s]
        rows.append({
            "Sector": s.title(),
            "Direct ($B)": round(d["direct"], 2),
            "Cascade ($B)": round(d["cascade"], 2),
            "Total ($B)": round(d["total"], 2),
            "Cascade share (%)": round(100.0 * d["cascade"] / d["total"], 1)
                                  if d["total"] > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def equity_frontier(weights: Optional[List[float]] = None,
                    target_count: int = 15,
                    scenario_class: str = "full_closure") -> pd.DataFrame:
    """Efficiency-equity frontier (MIP).

    Sweep the equity weight on vulnerable countries' priority sectors and report aggregate expected damage (at true penalties) vs. oil-access Gini. Higher weight diverts scarce oil to vulnerable economies: lower Gini at higher damage. scenario_class sets the disruption severity.
    """
    if weights is None:
        weights = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
    rows = []
    for wt in weights:
        # Distinct large model per weight: do not cache, release before the next.
        m = solve(ALL_POLICIES, equity_weight=wt, target_count=target_count,
                  scenario_class=scenario_class, cache=False)
        loss = sum(country_damage(m).values())
        g = gini(np.array(list(country_access(m).values())))
        rows.append({"Scenario": scenario_class,
                     "Equity weight": wt,
                     "Aggregate loss ($B)": round(loss, 1),
                     "Oil-access Gini": round(g, 4)})
        del m
        gc.collect()
    return pd.DataFrame(rows)


POLICY_DISPLAY = {p.pid: p.name for p in POLICIES}


def _enumerate_subsets(policies: Sequence[str]
                       ) -> List[Tuple[str, ...]]:
    """Return all 2^|policies| subsets as sorted tuples."""
    return [tuple(sorted(c))
            for r in range(len(policies) + 1)
            for c in combinations(policies, r)]


def solve_all_subsets(policies: Sequence[str] = ALL_POLICIES,
                      scenario_class: str = "full_closure",
                      target_count: int = 15,
                      duration_days: int = 100,
                      chokepoint: str = "CP_HORMUZ",
                      enable_cascade: bool = True,
                      n_jobs: int = 1,
                      verbose: bool = True
                      ) -> Dict[Tuple[str, ...], float]:
    """Solve the MIP for every subset of policies.

    Returns {subset (sorted tuple): expected-loss objective}; infeasible solves map to inf to keep the value function well-defined. solve_mip caches to disk.
    """
    subsets = _enumerate_subsets(policies)
    n = len(subsets)

    if verbose:
        print(f"Solving {n} MIP subset problems for {scenario_class}...")

    results: Dict[Tuple[str, ...], float] = {}

    def _solve_one(subset):
        r = solve_mip(policy_subset=subset,
                      scenario_class=scenario_class,
                      target_count=target_count,
                      duration_days=duration_days,
                      chokepoint=chokepoint,
                      enable_cascade=enable_cascade,
                      verbose=False)
        obj = r.objective if r.objective is not None else float("inf")
        return subset, obj, r.cached, r.runtime_sec

    if n_jobs == 1:
        for i, subset in enumerate(subsets, 1):
            subset, obj, cached, rt = _solve_one(subset)
            results[subset] = obj
            if verbose:
                tag = "cache" if cached else f"{rt:.1f}s"
                print(f"  [{i:2d}/{n}] {subset} -> {obj:,.1f}  ({tag})")
    else:
        try:
            from joblib import Parallel, delayed
            out = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(_solve_one)(s) for s in subsets)
            for subset, obj, _, _ in out:
                results[subset] = obj
        except ImportError:
            return solve_all_subsets(policies, scenario_class, target_count,
                                     duration_days, chokepoint,
                                     enable_cascade, n_jobs=1,
                                     verbose=verbose)

    return results


def _value_function(subset_objs: Dict[Tuple[str, ...], float]
                    ) -> Dict[Tuple[str, ...], float]:
    """Characteristic function v(S) = obj(empty) - obj(S), the loss reduction from S.

    v(empty) = 0 and v is monotone non-decreasing (adding policies cannot worsen the optimum).
    """
    baseline = subset_objs[tuple()]
    return {s: baseline - obj for s, obj in subset_objs.items()}


def _shapley_values(value_fn: Dict[Tuple[str, ...], float],
                    policies: Sequence[str]) -> Dict[str, float]:
    """Exact Shapley values phi_p = sum_S weight(S) * marginal(p, S)."""
    n = len(policies)
    n_fact = factorial(n)
    phi = {p: 0.0 for p in policies}

    for p in policies:
        others = [q for q in policies if q != p]
        for r in range(len(others) + 1):
            for S in combinations(others, r):
                S_sorted = tuple(sorted(S))
                S_with_p = tuple(sorted(S_sorted + (p,)))
                marginal = value_fn[S_with_p] - value_fn[S_sorted]
                weight = factorial(r) * factorial(n - r - 1) / n_fact
                phi[p] += weight * marginal
    return phi


def _sequential_attribution(value_fn: Dict[Tuple[str, ...], float],
                             order: Sequence[str]) -> Dict[str, float]:
    """Marginal value each policy adds as it joins the coalition in ``order``."""
    contrib: Dict[str, float] = {}
    coalition: List[str] = []
    for p in order:
        before = tuple(sorted(coalition))
        coalition.append(p)
        after = tuple(sorted(coalition))
        contrib[p] = value_fn[after] - value_fn[before]
    return contrib


def _leave_one_out(value_fn: Dict[Tuple[str, ...], float],
                    policies: Sequence[str]) -> Dict[str, float]:
    """LOO = v(full) - v(full minus {p}): how irreplaceable each policy is."""
    full = tuple(sorted(policies))
    return {p: value_fn[full] - value_fn[tuple(sorted(set(policies) - {p}))]
            for p in policies}


def attribution_table(scenario_class: str = "full_closure",
                       target_count: int = 15,
                       duration_days: int = 100,
                       chokepoint: str = "CP_HORMUZ",
                       enable_cascade: bool = True,
                       policies: Sequence[str] = ALL_POLICIES,
                       n_jobs: int = 1,
                       verbose: bool = True) -> pd.DataFrame:
    """Consolidated attribution table per instrument: Shapley ($B, %), sequential cheap-first and fast-first ($B), leave-one-out ($B, %).

    Dollar values are $B (MIP objective is $M, divided by 1000).
    """
    subset_objs = solve_all_subsets(
        policies=policies,
        scenario_class=scenario_class,
        target_count=target_count,
        duration_days=duration_days,
        chokepoint=chokepoint,
        enable_cascade=enable_cascade,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    value_fn = _value_function(subset_objs)   # loss reduction in $M
    total_saving = value_fn[tuple(sorted(policies))]

    if verbose:
        print(f"\nBaseline (no-policy) loss: {subset_objs[tuple()] / 1000:,.1f} $B")
        print(f"Full-portfolio loss:       "
              f"{subset_objs[tuple(sorted(policies))] / 1000:,.1f} $B")
        print(f"Total saving:              {total_saving / 1000:,.1f} $B "
              f"({total_saving / subset_objs[tuple()] * 100:.1f}%)")

    shapley = _shapley_values(value_fn, policies)

    # Cheap-first: ascending activation cost.
    cheap_order = sorted(policies, key=lambda p: POLICY_DICT[p].cost)
    cheap_seq = _sequential_attribution(value_fn, cheap_order)

    # Fast-first: reversible before irreversible, cost as tie-breaker.
    fast_order = sorted(policies,
                        key=lambda p: (POLICY_DICT[p].irreversible,
                                       POLICY_DICT[p].cost))
    fast_seq = _sequential_attribution(value_fn, fast_order)

    loo = _leave_one_out(value_fn, policies)

    # Round noise (|x| < 0.005) to 0.0 so inactive policies display clean zeros.
    def _clean(x: float) -> float:
        r = round(x, 2)
        return 0.0 if abs(r) < 0.005 else r

    rows = []
    for p in policies:
        rows.append({
            "Instrument": POLICY_DISPLAY[p],
            "Shapley ($B)":            _clean(shapley[p] / 1000),
            "Shapley (%)":             _clean(100 * shapley[p] / total_saving)
                                       if total_saving > 0 else 0.0,
            "Seq. cheap-first ($B)":   _clean(cheap_seq[p] / 1000),
            "Seq. fast-first ($B)":    _clean(fast_seq[p] / 1000),
            "LOO ($B)":                _clean(loo[p] / 1000),
            "LOO (%)":                 _clean(100 * loo[p] / total_saving)
                                       if total_saving > 0 else 0.0,
        })

    df = pd.DataFrame(rows)

    summary_row = {
        "Instrument": "Total (full portfolio)",
        "Shapley ($B)":            _clean(sum(shapley.values()) / 1000),
        "Shapley (%)":             _clean(100 * sum(shapley.values()) / total_saving)
                                   if total_saving > 0 else 0.0,
        "Seq. cheap-first ($B)":   _clean(sum(cheap_seq.values()) / 1000),
        "Seq. fast-first ($B)":    _clean(sum(fast_seq.values()) / 1000),
        "LOO ($B)":                _clean(sum(loo.values()) / 1000),
        "LOO (%)":                 _clean(100 * sum(loo.values()) / total_saving)
                                   if total_saving > 0 else 0.0,
    }
    df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

    return df


# Regime portfolios compared across scenario classes
REGIMES = {
    "No intervention":  (),
    "Reactive bundle":  ("P1", "P2", "P3"),
    "Full proactive":   ("P1", "P2", "P3", "P4", "P5", "P6"),
}


def regime_comparison(scenario_classes: Sequence[str] = (
                          "partial_restriction",
                          "full_closure",
                          "multi_chokepoint",
                      ),
                      target_count: int = 15,
                      enable_cascade: bool = True,
                      chokepoint: str = "CP_HORMUZ",
                      verbose: bool = True) -> pd.DataFrame:
    """Scenario-class by regime comparison (No intervention, Reactive, Full proactive, Saving %). Values in $B from the MIP objective.
    """
    labels = {
        "partial_restriction": "Partial restriction",
        "full_closure":        "Full closure",
        "multi_chokepoint":    "Multi-chokepoint",
        "attrition_mild":      "Attrition (mild)",
        "attrition_moderate":  "Attrition (moderate)",
        "attrition_severe":    "Attrition (severe)",
    }

    rows = []
    for sc in scenario_classes:
        row = {"Scenario": labels.get(sc, sc)}
        regime_values = {}
        for regime_name, subset in REGIMES.items():
            r = solve_mip(policy_subset=subset,
                          scenario_class=sc,
                          target_count=target_count,
                          chokepoint=chokepoint,
                          enable_cascade=enable_cascade,
                          verbose=False)
            obj_b = r.objective / 1000 if r.objective is not None else float("nan")
            regime_values[regime_name] = obj_b
            row[regime_name] = round(obj_b, 1)
        saving = (1.0 - regime_values["Full proactive"]
                  / regime_values["No intervention"]) * 100.0
        row["Saving vs. inaction"] = f"{saving:.1f}%"
        rows.append(row)
        if verbose:
            print(f"  {labels.get(sc, sc):<22}  "
                  f"no-int={row['No intervention']:>8.1f}  "
                  f"reactive={row['Reactive bundle']:>8.1f}  "
                  f"proactive={row['Full proactive']:>8.1f}  "
                  f"saving={row['Saving vs. inaction']}")

    return pd.DataFrame(rows)


def timing_delay_mip(scenario_class: str = "full_closure",
                      target_count: int = 15,
                      policy_subset: Sequence[str] = ALL_POLICIES,
                      chokepoint: str = "CP_HORMUZ",
                      enable_cascade: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """Solve the stochastic MIP under each delay tau = 0..NUM_STAGES, forcing z = 0 for stages 0..tau-1.

    Columns: Scenario, Delay (stages), Delay (days), Loss ($B), Vs immediate (%).
    """
    rows = []
    immediate_obj = None
    for tau in range(NUM_STAGES + 1):
        r = solve_mip(policy_subset=tuple(policy_subset),
                      scenario_class=scenario_class,
                      target_count=target_count,
                      chokepoint=chokepoint,
                      enable_cascade=enable_cascade,
                      delay_stages=tau,
                      verbose=False)
        if r.objective is None:
            obj_b = float("nan")
        else:
            obj_b = r.objective / 1000

        if tau == 0:
            immediate_obj = obj_b
            pct = "---"
            label = "Immediate (stage 1)"
        elif tau == NUM_STAGES:
            label = "Complete inaction"
            pct = (f"+{(obj_b / immediate_obj - 1) * 100:.1f}%"
                   if immediate_obj else "---")
        else:
            label = f"Delayed by {tau} stage{'s' if tau > 1 else ''} ({tau * STAGE_DAYS} days)"
            pct = (f"+{(obj_b / immediate_obj - 1) * 100:.1f}%"
                   if immediate_obj else "---")

        rows.append({
            "Activation timing":   label,
            "Delay (stages)":      tau,
            "Delay (days)":        tau * STAGE_DAYS,
            "Expected loss ($B)":  round(obj_b, 1),
            "Penalty vs immediate": pct,
        })
        if verbose:
            print(f"  {label}: obj={obj_b:,.1f} $B {pct}")

    return pd.DataFrame(rows)


def duration_sensitivity_mip(durations: Optional[List[int]] = None,
                              scenario_class: str = "full_closure",
                              target_count: int = 15,
                              chokepoint: str = "CP_HORMUZ",
                              enable_cascade: bool = True,
                              verbose: bool = True) -> pd.DataFrame:
    """Sweep disruption duration under three regimes: no intervention, reactive bundle, full proactive.

    The MIP keeps four stages but each stage length is duration_days / NUM_STAGES (may be fractional, so any horizon is exact). This rescales per-stage demand, reserve drawdown caps, and substitute caps proportionally, leaving tree, nonanticipativity, and activation logic unchanged. Stage decisions coarsen as stage length grows.
    """
    if durations is None:
        durations = [40, 80, 100, 200, 300, 365]

    regimes = {
        "No intervention": (),
        "Reactive bundle": ("P1", "P2", "P3"),
        "Full proactive":  ALL_POLICIES,
    }

    rows = []
    for d in durations:
        # NUM_STAGES equal stages of d/NUM_STAGES days (may be fractional, so the horizon is exact, e.g. 365 days = four 91.25-day stages).
        sd = d / NUM_STAGES
        horizon = NUM_STAGES * sd
        row = {"Duration (days)": int(round(horizon)), "Stage length (days)": round(sd, 2)}
        for name, subset in regimes.items():
            r = solve_mip(policy_subset=subset,
                          scenario_class=scenario_class,
                          target_count=target_count,
                          duration_days=horizon,
                          stage_days=sd,
                          chokepoint=chokepoint,
                          enable_cascade=enable_cascade,
                          verbose=False)
            obj_b = r.objective / 1000 if r.objective is not None else float("nan")
            row[name] = round(obj_b, 1)
        # z=0 is always feasible, so the optimized value weakly dominates inaction. On heavy long-horizon solves the time limit can leave a suboptimal incumbent above the attainable inaction value, so floor optimized at inaction. The reactive bundle is committed and may genuinely exceed inaction.
        if row.get("No intervention") and \
                row.get("Full proactive", float("inf")) > row["No intervention"]:
            row["Full proactive"] = row["No intervention"]
        saving = (1 - row["Full proactive"] / row["No intervention"]) * 100 \
                  if row["No intervention"] else float("nan")
        row["Saving vs. inaction"] = f"{saving:.1f}%"
        rows.append(row)
        if verbose:
            print(f"  duration={horizon}d (stage={sd}d): "
                  f"no-int={row['No intervention']:.1f} "
                  f"react={row['Reactive bundle']:.1f} "
                  f"proact={row['Full proactive']:.1f} "
                  f"saving={row['Saving vs. inaction']}")

    return pd.DataFrame(rows)


def _deterministic_scenario(path: List[float]) -> List[Scenario]:
    """Wrap a single availability path as a 1-scenario list with named state labels."""
    # Label each fraction by its nearest named state.
    avail_to_state = {v: k for k, v in STATE_AVAILABILITY.items()}
    states = []
    for a in path:
        closest = min(STATE_AVAILABILITY.items(),
                      key=lambda kv: abs(kv[1] - a))[0]
        states.append(closest)
    return [Scenario(omega=0, prob=1.0,
                     states=states, availability=path)]


# Wait-and-see: solve each scenario in isolation.
def wait_and_see(scenario_class: str = "full_closure",
                 target_count: int = 15,
                 chokepoint: str = "CP_HORMUZ",
                 enable_cascade: bool = True,
                 policy_subset: Sequence[str] = ALL_POLICIES,
                 verbose: bool = True) -> float:
    """Expected cost under perfect information (WS).

    Solve each scenario as its own MIP with the path known in advance (no non-anticipativity), then take the probability-weighted sum.
    """
    network = get_network()
    scenarios = get_scenarios(scenario_class, target_count)
    solver_name = _detect_solver()

    ws_value = 0.0
    if verbose:
        print(f"Wait-and-see: solving {len(scenarios)} single-scenario MIPs...")

    for w, sc in enumerate(scenarios):
        single = [Scenario(omega=0, prob=1.0,
                           states=sc.states, availability=sc.availability)]
        m = build_model(network, single,
                        disrupted_chokepoint=disrupted_chokepoints_for_class(
                            scenario_class, chokepoint),
                        enable_cascade=enable_cascade,
                        endogenous_uncertainty=False,
                        unresponsive_regime=class_regime(scenario_class)[1],
                        policy_subset=list(policy_subset))
        t0 = time.time()
        info = solve_model(m, solver_name=solver_name,
                           time_limit=600, mip_gap=1e-3)
        if info["objective"] is None:
            if verbose:
                print(f"  scenario {w}: INFEASIBLE")
            continue
        ws_value += sc.prob * info["objective"]
        if verbose:
            print(f"  scenario {w} (p={sc.prob:.4f}): "
                  f"obj={info['objective']:.1f}, t={time.time()-t0:.1f}s")

    return ws_value


# Expected-value problem.
def expected_value_problem(scenario_class: str = "full_closure",
                            target_count: int = 15,
                            chokepoint: str = "CP_HORMUZ",
                            enable_cascade: bool = True,
                            policy_subset: Sequence[str] = ALL_POLICIES,
                            verbose: bool = True
                            ) -> Tuple[float, Dict]:
    """Solve the deterministic problem at the mean availability path. Returns (objective, z_star)."""
    scenarios = get_scenarios(scenario_class, target_count)

    # Probability-weighted mean availability path.
    n_stages = len(scenarios[0].availability)
    mean_path = [
        sum(s.prob * s.availability[t] for s in scenarios) /
        sum(s.prob for s in scenarios)
        for t in range(n_stages)
    ]
    if verbose:
        print(f"EV problem: mean availability path = "
              f"{[round(x, 3) for x in mean_path]}")

    network = get_network()
    det = _deterministic_scenario(mean_path)
    m = build_model(network, det,
                    disrupted_chokepoint=disrupted_chokepoints_for_class(
                        scenario_class, chokepoint),
                    enable_cascade=enable_cascade,
                    endogenous_uncertainty=False,
                    unresponsive_regime=class_regime(scenario_class)[1],
                    policy_subset=list(policy_subset))
    solver_name = _detect_solver()
    info = solve_model(m, solver_name=solver_name,
                       time_limit=600, mip_gap=1e-3)

    if info["objective"] is None:
        return float("nan"), {}

    z_star = {}
    for p in policy_subset:
        for t in m.T:
            z_star[(p, t)] = int(round(pyo.value(m.z[p, t, 0])))

    if verbose:
        print(f"  EV objective: {info['objective']:.1f}")
        active = [(p, t) for (p, t), v in z_star.items() if v == 1]
        print(f"  EV policy plan: {active}")

    return info["objective"], z_star


# EEV: implement EV plan under stochastic realization.
def evaluate_ev_plan_stochastic(z_star: Dict[Tuple[str, int], int],
                                  scenario_class: str = "full_closure",
                                  target_count: int = 15,
                                  chokepoint: str = "CP_HORMUZ",
                                  enable_cascade: bool = True,
                                  policy_subset: Sequence[str] = ALL_POLICIES,
                                  verbose: bool = True) -> float:
    """Fix z = z* (the EV plan) and evaluate expected cost over the full scenario set (EEV)."""
    network = get_network()
    scenarios = get_scenarios(scenario_class, target_count)
    m = build_model(network, scenarios,
                    disrupted_chokepoint=disrupted_chokepoints_for_class(
                        scenario_class, chokepoint),
                    enable_cascade=enable_cascade,
                    endogenous_uncertainty=False,
                    unresponsive_regime=class_regime(scenario_class)[1],
                    policy_subset=list(policy_subset))

    # Pin z across every scenario to the EV plan.
    for p in policy_subset:
        for t in m.T:
            target = z_star.get((p, t), 0)
            for w in m.OMEGA:
                m.z[p, t, w].fix(target)

    solver_name = _detect_solver()
    t0 = time.time()
    info = solve_model(m, solver_name=solver_name,
                       time_limit=600, mip_gap=1e-3)
    if verbose:
        obj = "infeasible" if info["objective"] is None else f"{info['objective']:.1f}"
        print(f"  EEV objective: {obj}  (solve time {time.time()-t0:.1f}s)")
    return info["objective"] if info["objective"] is not None else float("inf")


# Top-level diagnostics.
def compute_diagnostics(scenario_classes: Sequence[str] = (
                            "partial_restriction",
                            "full_closure",
                            "multi_chokepoint",
                        ),
                        target_count: int = 15,
                        chokepoint: str = "CP_HORMUZ",
                        enable_cascade: bool = True,
                        policy_subset: Sequence[str] = ALL_POLICIES,
                        verbose: bool = True) -> pd.DataFrame:
    """EVPI / VSS diagnostics across scenario classes.

    Columns ($B): RP (stochastic optimum), WS (wait-and-see), EEV (EV plan under realization), EVPI = RP - WS, VSS = EEV - RP, each also as % of RP.
    """
    rows = []
    for sc in scenario_classes:
        if verbose:
            print(f"\n=== {sc} ===")

        # RP on the exogenous model so the EVPI/VSS comparison is internally consistent (WS, EV, EEV all exogenous). Endogenous-uncertainty value is reported elsewhere.
        rp_m = build_model(
            get_network(), get_scenarios(sc, target_count),
            disrupted_chokepoint=disrupted_chokepoints_for_class(sc, chokepoint),
            enable_cascade=enable_cascade, endogenous_uncertainty=False,
            unresponsive_regime=class_regime(sc)[1],
            policy_subset=list(policy_subset))
        rp_info = solve_model(rp_m, solver_name=_detect_solver(),
                              time_limit=600, mip_gap=1e-3)
        rp_obj = rp_info["objective"]
        if verbose:
            print(f"RP (stochastic optimum): {rp_obj:.1f}")

        # WS
        ws_obj = wait_and_see(scenario_class=sc,
                              target_count=target_count,
                              chokepoint=chokepoint,
                              enable_cascade=enable_cascade,
                              policy_subset=policy_subset,
                              verbose=verbose)

        # EV problem -> z*
        ev_obj, z_star = expected_value_problem(scenario_class=sc,
                                                 target_count=target_count,
                                                 chokepoint=chokepoint,
                                                 enable_cascade=enable_cascade,
                                                 policy_subset=policy_subset,
                                                 verbose=verbose)

        # EEV
        eev_obj = evaluate_ev_plan_stochastic(z_star,
                                                scenario_class=sc,
                                                target_count=target_count,
                                                chokepoint=chokepoint,
                                                enable_cascade=enable_cascade,
                                                policy_subset=policy_subset,
                                                verbose=verbose)

        evpi = rp_obj - ws_obj
        vss = eev_obj - rp_obj

        rows.append({
            "Scenario class": sc,
            "RP ($B)":   round(rp_obj / 1000, 1),
            "WS ($B)":   round(ws_obj / 1000, 1),
            "EEV ($B)":  round(eev_obj / 1000, 1) if np.isfinite(eev_obj) else float("nan"),
            "EVPI ($B)": round(evpi / 1000, 1),
            "VSS ($B)":  round(vss / 1000, 1) if np.isfinite(vss) else float("nan"),
            "EVPI (% of RP)": round(100 * evpi / rp_obj, 1),
            "VSS (% of RP)":  round(100 * vss / rp_obj, 1) if np.isfinite(vss) else float("nan"),
        })

    return pd.DataFrame(rows)


# Core loss computation (reduced-form screen)
def compute_system_losses(
    countries: Dict[str, Country],
    availability: float = 0.0,
    duration_days: int = 100,
    policy_reduction: float = 0.0,
    chokepoint_dep_attr: str = "hormuz_dep",
) -> pd.DataFrame:
    """Per-country economic losses for a disruption scenario and policy level.

    availability: fraction of chokepoint capacity remaining (0 = full closure). policy_reduction: fractional loss reduction from active policies. chokepoint_dep_attr: country attribute name for chokepoint dependence.
    """
    rows = []
    for c in countries.values():
        dep = getattr(c, chokepoint_dep_attr, c.hormuz_dep)
        oil_red = dep * (1.0 - availability)
        spr_offset = min(oil_red,
                         (c.spr_days / max(duration_days, 1)) * oil_red)
        net_red = max(0.0, oil_red - spr_offset)

        fulfillment = {k: max(0.0, 1.0 - net_red * w)
                       for k, w in c.sector_weights.items()}
        cascade_out = cascade_single_pass(fulfillment)
        output_loss = sum((1.0 - cascade_out[k]) * c.sector_weights.get(k, 0.0)
                          for k in SECTORS)

        daily_gdp = c.gdp_B / 365.0
        raw_loss = daily_gdp * output_loss * duration_days * LOSS_SCALING
        adj_loss = raw_loss * (1.0 - policy_reduction)

        rows.append({
            "Country": c.name, "Code": c.code, "Region": c.region,
            "GDP ($B)": c.gdp_B, "Demand (mbd)": c.daily_oil_mbd,
            "Dep.": dep, "SPR (days)": c.spr_days,
            "Net Oil Red.": net_red,
            "Output Loss": output_loss,
            "Econ. Loss ($B)": round(adj_loss, 2),
        })

    return (pd.DataFrame(rows)
            .sort_values("Econ. Loss ($B)", ascending=False)
            .reset_index(drop=True))


# ESCVI: Energy Supply Chain Vulnerability Index
def compute_escvi(
    countries: Dict[str, Country],
    chokepoint: str = "hormuz",
) -> pd.DataFrame:
    """Composite vulnerability index (import dependence, reserve vulnerability, alternative-supply gap, cascade severity), ranked. chokepoint selects the dependence attribute.
    """
    dep_attr = f"{chokepoint}_dep"
    rows = []

    for c in countries.values():
        # National exposure delta_n = min(1, iota_n * theta_n), the same quantity the optimization uses (import dependence times route share, capped at one).
        dep = min(1.0, getattr(c, dep_attr, c.hormuz_dep) * IMPORT_SHARE.get(c.code, 1.0))

        import_dep = dep * 100.0
        reserve_vuln = max(0.0, (1.0 - c.spr_days / 90.0)) * 100.0
        # Alternative-supply gap: exposed demand net of the alternatives that blunt it (renewable penetration plus the ~5%-of-demand substitute capability: product stocks, refinery reallocation, fuel switching). Reserve buffering is captured separately.
        alt_capability = c.renew_pct / 100.0 + SUBSTITUTE_ALT_FRAC
        alt_gap = max(0.0, 1.0 - alt_capability) * dep * 100.0

        cascade_res = compute_cascade_losses(c, dep, 100)
        summary = cascade_res.get("_summary", {})
        cascade_sev = summary.get("weighted_output_loss", 0.0) * 100.0

        escvi = (0.30 * import_dep + 0.30 * reserve_vuln +
                 0.20 * alt_gap + 0.20 * cascade_sev)

        rows.append({
            "Country": c.name, "Code": c.code, "Region": c.region,
            "ESCVI": round(escvi, 1),
            "Import Dep.": round(import_dep, 1),
            "Reserve Vuln.": round(reserve_vuln, 1),
            "Alt. Gap": round(alt_gap, 1),
            "Cascade Sev.": round(cascade_sev, 1),
        })

    return (pd.DataFrame(rows)
            .sort_values("ESCVI", ascending=False)
            .reset_index(drop=True))


def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array of values."""
    values = np.array(values, dtype=float)
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return (2.0 * np.sum(index * sorted_v) / (n * np.sum(sorted_v))) - (n + 1) / n


# Model variant comparison
# Cache keyed by (enable_cascade, endogenous, optimized, chokepoint, duration_days).
_variant_solve_cache: Dict[Tuple[bool, bool, bool, str, int], float] = {}


def _solve_variant_objective(
    enable_cascade: bool,
    endogenous: bool,
    optimized: bool,
    chokepoint: str,
    duration_days: int,
    scenario_class: str = "full_closure",
    target_count: int = 15,
) -> float:
    """Build and solve one model formulation, returning expected system loss (USD billion).

    enable_cascade: include the cascade penalty. endogenous: policy actions shape transition probabilities (action-contingent uncertainty), else exogenous path. optimized: full six-instrument portfolio vs empty subset.
    """

    key = (enable_cascade, endogenous, optimized, chokepoint, duration_days)
    if key in _variant_solve_cache:
        return _variant_solve_cache[key]

    network = get_network()
    scenarios = get_scenarios(scenario_class, target_count)
    policy_subset = ["P1", "P2", "P3", "P4", "P5", "P6"] if optimized else []

    model = build_model(
        network, scenarios,
        disrupted_chokepoint=chokepoint,
        duration_days=duration_days,
        enable_cascade=enable_cascade,
        endogenous_uncertainty=endogenous,
        unresponsive_regime=class_regime(scenario_class)[1],
        policy_subset=policy_subset,
    )
    solver_name = _detect_solver(None)
    info = solve_model(model, solver_name=solver_name)
    if info["objective"] is None:
        raise RuntimeError(
            f"Model-variant solve infeasible (cascade={enable_cascade}, "
            f"endogenous={endogenous}, optimized={optimized})"
        )
    obj = float(info["objective"])
    obj_billion = obj / 1000.0   # USD million to USD billion
    _variant_solve_cache[key] = obj_billion
    return obj_billion


def model_variant_comparison(
    countries: Dict[str, Country],
    chokepoint: str = "CP_HORMUZ",
    duration_days: int = 100,
) -> pd.DataFrame:
    """Compare expected system losses with and without the cascade module, isolating its value.

    Both rows are the same stochastic program, differing only in whether the cascade penalty is active (off: unmet demand penalized by direct linear shortage cost only). Both are genuine solves, so the gap measures the value of modeling cascades.
    """
    variants = [
        ("Direct shortage only (no cascade)", False),
        ("With cross-sector cascade",         True),
    ]

    rows = []
    for label, casc in variants:
        l_none = _solve_variant_objective(
            casc, True, optimized=False,
            chokepoint=chokepoint, duration_days=duration_days,
        )
        l_opt = _solve_variant_objective(
            casc, True, optimized=True,
            chokepoint=chokepoint, duration_days=duration_days,
        )
        rows.append({
            "Model Variant": label,
            "No Intervention ($B)": round(l_none, 1),
            "Optimized ($B)": round(l_opt, 1),
        })
    return pd.DataFrame(rows)


def amplification_pwl_error(m=None) -> pd.DataFrame:
    """PWL under-estimation error for both the amplification damage g_s(u) and the world-price deadweight.

    CERTIFIED is the worst-case relative gap between the true convex function and its max-tangent epigraph on a dense grid. REALIZED (if model m given) is the prob-weighted gap at the optimum. Both one-sided (epigraph below the curve, so the MIP under-counts).
    """
    grid = np.linspace(0.0, 1.0, 2001)
    rows = []
    # Relative error reported over the meaningful range (g(u) >= 1% of g(1)); below that damage and gap are negligible and the relative gap degenerates as g(u) vanishes near u=0. Certified absolute gap (a true one-sided under-count bound) reported alongside.
    for s in SECTORS:
        segs = build_pwl_segments(s)
        true = np.array([damage_value(s, u) for u in grid])
        approx = np.array([max(a * u + b for a, b in segs) for u in grid])
        abs_gap = np.max(true - approx)
        meaningful = true > 0.01 * true[-1]
        rel = float(np.max((true[meaningful] - approx[meaningful]) / true[meaningful])) * 100.0
        rows.append({"Function": f"amplification g_{s}",
                     "Certified max rel. error (%)": round(rel, 2),
                     "Certified max abs. gap": round(float(abs_gap), 4)})
    # Deadweight: quadratic dQ^2 vs its tangents on [0, dq_max], scale-free in c.
    nbp = 12
    q = np.linspace(0.0, 1.0, 2001)
    bp = [k / nbp for k in range(nbp + 1)]
    true_d = q ** 2
    approx_d = np.array([max(2.0 * qk * qi - qk ** 2 for qk in bp) for qi in q])
    meaningful_d = true_d > 0.01 * true_d[-1]
    rel_d = float(np.max((true_d[meaningful_d] - approx_d[meaningful_d])
                         / true_d[meaningful_d])) * 100.0
    rows.append({"Function": "deadweight (dQ)^2",
                 "Certified max rel. error (%)": round(rel_d, 2),
                 "Certified max abs. gap": round(float(np.max(true_d - approx_d)), 4)})
    df = pd.DataFrame(rows)
    if m is not None:
        casc_true = _cascade_damage_of_solution(m)
        casc_lin = cost_components(m).get("Cascade", 0.0)
        realized = 100.0 * (casc_true - casc_lin) / max(casc_true, 1e-9)
        df.attrs["realized_amplification_error_pct"] = round(realized, 2)
        print(f"  realized amplification error at optimum = {realized:.2f}% "
              f"(true {casc_true:.2f} vs linearized {casc_lin:.2f})")
    return df


def _cascade_damage_of_solution(m) -> float:
    """Cascade damage (USD billion) implied by a solved allocation's shortfalls, recomputed from BETA and the convex damage g_s(u).

    Works without the cascade module (no m.Delta), so a naive allocation can be scored under the true objective. At the cascade optimum the epigraph binds, reproducing cost_components["Cascade"]. Used by the four-cell cross-evaluation.
    """
    pw = realized_scenario_probs(m)
    total = 0.0
    for w in m.OMEGA:
        for t in m.T:
            for n in m.C:
                rscale = pyo.value(m.region_cascade_scale[n]) \
                    if hasattr(m, "region_cascade_scale") else 1.0
                for s in m.S:
                    u = 0.0
                    for sp, beta_val in BETA.get(s, {}).items():
                        d = pyo.value(m.demand[n, sp, t])
                        if d > 1e-10:
                            u += rscale * beta_val * (pyo.value(m.h[n, sp, t, w]) / d
                                             + (1.0 - EFFICIENCY_FRAC)
                                             * pyo.value(m.rho[n, sp, t, w]))
                    if u <= 0.0:
                        continue
                    total += (pw[w] * pyo.value(m.equity_mult[n, s])
                              * pyo.value(m.eta_ns[n, s]) * _cscale(m, n, s, t)
                              * damage_value(s, u))
    return total / 1e3


def cascade_cross_evaluation(chokepoint: str = "CP_HORMUZ",
                             duration_days: int = 100,
                             scenario_class: str = "full_closure") -> pd.DataFrame:
    """Four-cell cross-evaluation of the cascade module.

    Each model's SOLUTION is scored under BOTH objectives, avoiding the conflation of a naive-vs-cascade optimum difference:

        A  naive solution,   evaluated without cascade (its own objective)
        B  naive solution,   evaluated WITH cascade damage
        C  aware solution,   evaluated without cascade (direct cost only)
        D  aware solution,   evaluated with cascade (its own objective)

    Decomposes into pure cascade damage (B - A), allocation/policy gain from anticipating it (B - D), and the direct-cost trade-off the aware model accepts (C - A). Both regimes.
    """
    rows = []
    for regime, subset in [("No intervention", []), ("Optimized", list(ALL_POLICIES))]:
        m_nc = solve(subset, scenario_class=scenario_class, chokepoint=chokepoint,
                     duration_days=duration_days, enable_cascade=False, cache=False)
        A = cost_components(m_nc)["Total"]
        B = A + _cascade_damage_of_solution(m_nc)
        del m_nc
        gc.collect()
        m_c = solve(subset, scenario_class=scenario_class, chokepoint=chokepoint,
                    duration_days=duration_days, enable_cascade=True, cache=False)
        cc = cost_components(m_c)
        # C and D score the aware allocation with the same true damage g(u) used for B, so the derived quantities are consistent (linearized Delta differs from g(u) only by the small PWL gap).
        C = cc["Total"] - cc["Cascade"]
        D = C + _cascade_damage_of_solution(m_c)
        del m_c
        gc.collect()
        rows.append({
            "Regime": regime,
            "A naive/no-casc": round(A, 1),
            "B naive/casc": round(B, 1),
            "C aware/direct": round(C, 1),
            "D aware/casc": round(D, 1),
            "Pure cascade (B-A)": round(B - A, 1),
            "Allocation gain (B-D)": round(B - D, 1),
            "Direct trade-off (C-A)": round(C - A, 1),
        })
        print(f"  {regime:16s}  A={A:6.1f} B={B:6.1f} C={C:6.1f} D={D:6.1f}  "
              f"pure-cascade={B-A:5.1f}  alloc-gain={B-D:5.1f}  direct-tradeoff={C-A:5.1f}")
    return pd.DataFrame(rows)


_DEP_ATTR_BY_CP = {"CP_HORMUZ": "hormuz_dep", "CP_MALACCA": "malacca_dep",
                   "CP_SUEZ": "suez_dep"}

# Regional blocs for cooperative-game cost-sharing, matched by country code; unlisted importers fall into "Rest".
REGIONAL_BLOCS = {
    "East Asia": ["CHN", "JPN", "KOR", "TWN", "HKG"],
    "South Asia": ["IND", "PAK", "BGD", "LKA", "NPL"],
    "SE Asia": ["SGP", "THA", "IDN", "MYS", "VNM", "PHL", "MMR", "KHM"],
    "Europe": ["DEU", "ITA", "ESP", "FRA", "GBR", "NLD", "BEL", "POL", "GRC",
               "PRT", "SWE", "AUT", "CHE", "CZE", "ROU", "TUR"],
    "Africa": ["EGY", "ETH", "KEN", "UGA", "DJI", "NGA", "COD", "ZAF", "MAR",
               "TZA", "SDN", "GHA"],
}


def _bloc_of(code: str) -> str:
    for bloc, members in REGIONAL_BLOCS.items():
        if code in members:
            return bloc
    return "Rest"


def _price_channel_terms(m):
    """Physical price-channel deadweight and terms-of-trade transfer (USD billion), on the physical residual price_loss_phys.

    price_loss_phys nets out every importer's realized conservation (a managed reduction lowers world demand regardless of who chose it), unlike the objective's incentive residual price_loss (coordinating set only) that distinguishes price-taker from coordinator. For the planner the two coincide. Deadweight read off the same PWL envelope the objective uses.
    """
    pw = realized_scenario_probs(m)
    ws = float(WORLD_OIL_SUPPLY_MBD * STAGE_DAYS)
    bp = list(getattr(m, "_dwl_bp", []))
    curve = getattr(m, "_price_curve", "linear")
    eps = 1.0 / abs(OIL_DEMAND_ELASTICITY)
    dwl = transfer = 0.0
    for t in m.T:
        pc = pyo.value(m.price_coef[t])
        c = pyo.value(m.dwl_coef[t])
        Q0 = sum(pyo.value(m.demand[n, s, t]) for n in m.C for s in m.S)
        for w in m.OMEGA:
            pl = pyo.value(m.price_loss_phys[t, w])
            if curve == "isoelastic":
                f = min(pl / ws, 0.95)
                dwl += pw[w] * BASELINE_OIL_PRICE * Q0 * ((1.0 - (1.0 - f) ** (1.0 - eps)) / (1.0 - eps) - f) / 1e3
                transfer += pw[w] * BASELINE_OIL_PRICE * ((1.0 - f) ** (-eps) - 1.0) * Q0 * (1.0 - f) / 1e3
            else:
                dpw = max([0.0] + [2.0 * c * q * pl - c * q * q for q in bp]) if bp else c * pl * pl
                dwl += pw[w] * dpw / 1e3
                transfer += pw[w] * pc * pl * max(0.0, 1.0 - pl / ws) / 1e3
    return dwl, transfer


def _coordination_bundle(m) -> Dict[str, float]:
    """Welfare bundle for the decentralization analysis: real-resource efficiency loss, terms-of-trade transfer to producers, importer-coalition welfare (efficiency + transfer, the monopsony-relevant measure), and priced response cost (curtailment + reserve + substitute).
    """
    cc = cost_components(m)
    dwl, transfer = _price_channel_terms(m)
    efficiency = cc["Total"] - cc["Oil price"] + dwl
    dt = dwl + transfer                                     # deadweight + transfer
    response = cc["Curtailment"] + cc["Reserve"] + cc["Substitute"]
    return {"efficiency": efficiency, "transfer": transfer,
            "deadweight": dwl, "importer": efficiency + transfer,
            "dt": dt, "response": response}


def _bloc_response_costs(m) -> Dict[str, float]:
    """Per-bloc priced response outlays (curtailment + reserve + substitute, $B) from a solved model, for the cost-sharing transfer table.
    """
    pw = realized_scenario_probs(m)
    has_g = hasattr(m, "g_ps")
    out: Dict[str, float] = {}
    for n in m.C:
        bloc = _bloc_of(n)
        curt = sum(pw[w] * (EFFICIENCY_COST_FRAC * BASELINE_OIL_PRICE * EFFICIENCY_FRAC
                            + pyo.value(m.equity_mult[n, s]) * pyo.value(m.pi_ns[n, s])
                            * (1.0 - EFFICIENCY_FRAC))
                   * pyo.value(m.rho[n, s, t, w]) * pyo.value(m.demand[n, s, t])
                   for s in m.S for t in m.T for w in m.OMEGA) / 1e3
        res = (sum(pw[w] * RESERVE_REPLACEMENT_COST * pyo.value(m.r[n, t, w])
                   for t in m.T for w in m.OMEGA) / 1e3) if hasattr(m, "r") else 0.0
        sub = (sum(pw[w] * (SUBSTITUTE_COST_PS * pyo.value(m.g_ps[n, t, w])
                            + SUBSTITUTE_COST_RF * pyo.value(m.g_rf[n, t, w])
                            + SUBSTITUTE_COST_FS * pyo.value(m.g_fs[n, t, w]))
                   for t in m.T for w in m.OMEGA) / 1e3) if has_g else 0.0
        out[bloc] = out.get(bloc, 0.0) + curt + res + sub
    return out


def _realized_bloc_shares(m) -> Dict[str, float]:
    """Realized consumption incidence by bloc: prob-weighted served consumption as a share of the world total.

    Incidence base for the price benefit in place of baseline demand shares, so the benefit is attributed on crude actually cleared after the disruption, not pre-disruption consumption.
    """
    pw = realized_scenario_probs(m)
    cons: Dict[str, float] = {}
    for n in m.C:
        b = _bloc_of(n)
        cons[b] = cons.get(b, 0.0) + sum(
            pw[w] * pyo.value(m.q[n, s, t, w])
            for s in m.S for t in m.T for w in m.OMEGA)
    tot = sum(cons.values())
    return {b: v / tot for b, v in cons.items()} if tot > 1e-9 else {}


def coalition_transfer_table(shapley: Dict[str, float],
                             chokepoint: str = "CP_HORMUZ", duration_days: int = 100,
                             scenario_class: str = "full_closure",
                             target_count: int = 30) -> pd.DataFrame:
    """Budget-balanced cost-sharing table.

    Per bloc: gross price benefit (its consumption share of the world-price cost saved), extra response cost its conservation adds under coordination (grand minus noncoop), raw net, Shapley allocation, and the reconciling side transfer. Transfers sum to zero and net allocations sum to v(N). Needs the Shapley allocations from coalition_cost_sharing.
    """
    net = get_network()
    scen = get_scenarios(scenario_class, target_count)
    disr = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    unresp = class_regime(scenario_class)[1]

    def _solve(coord):
        m = build_model(net, scen, disrupted_chokepoint=disr, duration_days=duration_days,
                        unresponsive_regime=unresp, policy_subset=list(ALL_POLICIES),
                        coordinated_price=coord)
        solve_model(m, solver_name=_detect_solver(None), time_limit=600, mip_gap=1e-3)
        return m

    m_nc, m_gr = _solve(False), _solve(True)
    dt_saved = _coordination_bundle(m_nc)["dt"] - _coordination_bundle(m_gr)["dt"]
    resp_nc, resp_gr = _bloc_response_costs(m_nc), _bloc_response_costs(m_gr)

    blocs = list(REGIONAL_BLOCS.keys()) + ["Rest"]
    demand = {c: net.countries[c].daily_oil_mbd for c in net.countries}
    tot = sum(demand.values())
    base_share = {b: sum(demand[c] for c in net.countries if _bloc_of(c) == b) / tot for b in blocs}
    real_share = _realized_bloc_shares(m_nc)   # realized incidence
    share = {b: real_share.get(b, base_share[b]) for b in blocs}

    # Total real resource cost of coordination W'_increase = DT_saved - v(N); exceeds priced outlays by the efficiency distortion of holding the price down. Attributed by consumption share so raw net sums to v(N) and transfers net to zero. Priced outlays kept as a memo.
    coord_cost_total = dt_saved - sum(shapley.values())
    rows = []
    for b in blocs:
        gross = share[b] * dt_saved
        cost = share[b] * coord_cost_total
        raw_net = gross - cost
        net_alloc = shapley.get(b, 0.0)
        rows.append({"Bloc": b, "Consumption share (%)": round(100 * share[b], 1),
                     "Gross benefit ($B)": round(gross, 2),
                     "Coordination cost ($B)": round(cost, 2),
                     "Raw net ($B)": round(raw_net, 2),
                     "Shapley net ($B)": round(net_alloc, 2),
                     "Transfer ($B)": round(net_alloc - raw_net, 2),
                     "Priced outlay memo ($B)": round(resp_gr.get(b, 0.0) - resp_nc.get(b, 0.0), 2)})
    df = pd.DataFrame(rows)
    df.attrs["dt_saved"] = round(float(dt_saved), 2)
    df.attrs["coord_cost_total"] = round(float(coord_cost_total), 2)
    df.attrs["transfer_sum"] = round(float(df["Transfer ($B)"].sum()), 3)
    df.attrs["shapley_sum"] = round(float(df["Shapley net ($B)"].sum()), 2)
    return df


def coordination_analysis(chokepoint: str = "CP_HORMUZ", duration_days: int = 100,
                          scenario_class: str = "full_closure",
                          target_count: int = 15) -> pd.DataFrame:
    """Value of coordination: planner (importers internalize that aggregate restraint moderates the world price) versus the noncooperative price-taking benchmark.

    Reports the importer-coalition (monopsony) gain, Total + transfer, decomposed into the terms-of-trade transfer saved from producers and the global-efficiency effect. Coordination is privately valuable to importers even where the efficiency component is small or slightly negative (the buyer-monopsony result).
    """
    net = get_network()
    scen = get_scenarios(scenario_class, target_count)
    disr = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    unresp = class_regime(scenario_class)[1]

    def _solve(coord):
        m = build_model(net, scen, disrupted_chokepoint=disr, duration_days=duration_days,
                        unresponsive_regime=unresp, policy_subset=list(ALL_POLICIES),
                        coordinated_price=coord)
        solve_model(m, solver_name=_detect_solver(None), time_limit=600, mip_gap=1e-3)
        b = _coordination_bundle(m)
        del m
        gc.collect()
        return b

    plan = _solve(True)
    nc = _solve(False)
    rows = [
        {"Measure": "Importer-coalition welfare loss (Total + transfer)",
         "Noncooperative": round(nc["importer"], 1), "Planner": round(plan["importer"], 1),
         "Coordination gain": round(nc["importer"] - plan["importer"], 1)},
        {"Measure": "  Terms-of-trade transfer to producers",
         "Noncooperative": round(nc["transfer"], 1), "Planner": round(plan["transfer"], 1),
         "Coordination gain": round(nc["transfer"] - plan["transfer"], 1)},
        {"Measure": "  Global-efficiency welfare loss (deadweight-based)",
         "Noncooperative": round(nc["efficiency"], 1), "Planner": round(plan["efficiency"], 1),
         "Coordination gain": round(nc["efficiency"] - plan["efficiency"], 1)},
        {"Measure": "  Price deadweight",
         "Noncooperative": round(nc["deadweight"], 2), "Planner": round(plan["deadweight"], 2),
         "Coordination gain": round(nc["deadweight"] - plan["deadweight"], 2)},
    ]
    df = pd.DataFrame(rows)
    print(df.to_string(index=False), flush=True)
    print(f"  monopsony gain {nc['importer']-plan['importer']:.1f}B = transfer saved "
          f"{nc['transfer']-plan['transfer']:.1f}B - efficiency cost "
          f"{plan['efficiency']-nc['efficiency']:.1f}B", flush=True)
    return df


def coalition_cost_sharing(chokepoint: str = "CP_HORMUZ", duration_days: int = 100,
                           scenario_class: str = "full_closure",
                           target_count: int = 9, price_curve: str = "linear",
                           mip_gap: float = 1e-3, time_limit: int = 600) -> pd.DataFrame:
    """Cooperative-game cost-sharing of the coordination gain across regional blocs.

    Coalition value v(S) = share_S * (DT_noncoop - DT_S) - (Wprime_S - Wprime_noncoop), where DT is the world-price cost (deadweight + transfer) and Wprime is the real resource loss excluding it, so v(N) equals the planner-vs-lower-bound importer-welfare gain. Only S coordinates (non-S free-ride on the moderated price). Reports each bloc's Shapley allocation of v(N), individual rationality (phi_i >= v({i})), and core stability. Price-benefit leakage to free-riders makes small coalitions unprofitable, so the monopsony is under-provided without full coordination.
    """
    net = get_network()
    scen = get_scenarios(scenario_class, target_count)
    disr = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    unresp = class_regime(scenario_class)[1]

    blocs = list(REGIONAL_BLOCS.keys()) + ["Rest"]
    members = {b: [c for c in net.countries if _bloc_of(c) == b] for b in blocs}
    demand = {c: net.countries[c].daily_oil_mbd for c in net.countries}
    tot_demand = sum(demand.values())
    base_share = {b: sum(demand[c] for c in members[b]) / tot_demand for b in blocs}

    solve_gaps: List[float] = []

    def _solve(coord_codes, want_shares=False):
        m = build_model(net, scen, disrupted_chokepoint=disr, duration_days=duration_days,
                        unresponsive_regime=unresp, policy_subset=list(ALL_POLICIES),
                        coordinated_price=bool(coord_codes) or coord_codes == "ALL",
                        coordinating_countries=(None if coord_codes == "ALL" else coord_codes),
                        price_curve=price_curve)
        res = solve_model(m, solver_name=_detect_solver(None), time_limit=time_limit,
                          mip_gap=mip_gap)
        if res.get("gap") is not None:
            solve_gaps.append(float(res["gap"]))
        b = _coordination_bundle(m)
        b["_obj"] = res.get("objective")
        sh = _realized_bloc_shares(m) if want_shares else None
        del m
        gc.collect()
        return (b, sh) if want_shares else b

    nc, real_share = _solve([], want_shares=True)   # grand price-taking lower-bound baseline
    # Realized incidence: attribute the price benefit on crude each bloc actually clears, not baseline demand.
    share = {b: real_share.get(b, base_share[b]) for b in blocs}
    dt_nc = nc["dt"]
    # Real resource loss excluding the shared price deadweight (shortage + amplification + transport + response). A coalition bears its full increase from its own conservation but captures only its consumption share of the price benefit. Defined so v(N) = (dt_nc - dt_N) - (Wprime_N - Wprime_nc).
    wprime_nc = nc["efficiency"] - nc["deadweight"]

    value_fn: Dict[Tuple[str, ...], float] = {(): 0.0}
    for r in range(1, len(blocs) + 1):
        for S in combinations(blocs, r):
            codes = [c for b in S for c in members[b]]
            b = _solve(codes)
            share_S = sum(share[x] for x in S)
            wprime_S = b["efficiency"] - b["deadweight"]
            v_raw = share_S * (dt_nc - b["dt"]) - (wprime_S - wprime_nc)
            # Normalized game: any coalition can decline to coordinate and secure zero, so v_+(S) = max{0, v(S)} for every S (not only singletons). v(N) is positive so it is unaffected; unprofitable coalitions floored at their zero outside option before Shapley and core.
            v = max(0.0, v_raw)
            value_fn[tuple(sorted(S))] = v
            print(f"  v({'+'.join(S)}) = {v:.2f}B  (raw {v_raw:.2f})", flush=True)

    phi = _shapley_values(value_fn, blocs)
    vN = value_fn[tuple(sorted(blocs))]
    rows = []
    for b in blocs:
        solo = value_fn[(b,)]
        # Individual rationality against the outside option max{0, v({i})}: a negative stand-alone value does not lower the participation threshold.
        rows.append({"Bloc": b, "Consumption share (%)": round(100 * share[b], 1),
                     "Shapley allocation ($B)": round(phi[b], 2),
                     "Stand-alone v({i}) ($B)": round(solo, 2),
                     "Individually rational": phi[b] >= max(0.0, solo) - 1e-6})
    # Core check: the allocation is in the core iff excess sum(phi_S) - v(S) >= 0 for every S. Grand-coalition excess is zero by efficiency, so report the minimum over proper coalitions; strictly positive proper slack places the allocation in the relative interior.
    all_excess = {S: sum(phi[b] for b in S) - value_fn[tuple(sorted(S))]
                  for r in range(1, len(blocs) + 1) for S in combinations(blocs, r)}
    core_ok = min(all_excess.values()) >= -1e-6
    proper = {S: e for S, e in all_excess.items() if len(S) < len(blocs)}
    worst_S = min(proper, key=proper.get)
    min_proper_excess = proper[worst_S]
    # Certification: each coalition solves to relative gap <= mip_gap, so a world-price cost of scale |dt_nc| carries objective uncertainty ~ max_gap * |dt_nc|. v(S), Shapley, and excess are linear in these, so flag min proper excess as certified positive only when it comfortably exceeds that scale.
    max_gap = max(solve_gaps) if solve_gaps else float("nan")
    obj_uncertainty = max_gap * abs(dt_nc) if solve_gaps else float("nan")
    certified_positive = bool(min_proper_excess > 10.0 * obj_uncertainty) if solve_gaps else False
    df = pd.DataFrame(rows)
    df.attrs["grand_coalition_value"] = round(vN, 2)
    df.attrs["shapley_sum"] = round(sum(phi.values()), 2)
    df.attrs["core_stable"] = bool(core_ok)
    df.attrs["min_proper_core_excess"] = round(float(min_proper_excess), 3)
    df.attrs["closest_blocking_coalition"] = "+".join(worst_S)
    df.attrs["mip_gap_target"] = mip_gap
    df.attrs["max_achieved_gap"] = round(float(max_gap), 6)
    df.attrs["obj_uncertainty_B"] = round(float(obj_uncertainty), 4)
    df.attrs["core_excess_certified_positive"] = certified_positive
    print(f"  v(N)={vN:.2f}B  Shapley sum={sum(phi.values()):.2f}B  "
          f"min proper-coalition excess={min_proper_excess:.2f}B at {'+'.join(worst_S)}  "
          f"core stable={core_ok}", flush=True)
    print(f"  [certify] max achieved gap={max_gap:.2e}  "
          f"obj uncertainty~{obj_uncertainty:.4f}B  "
          f"min excess certified strictly positive={certified_positive}", flush=True)
    return df


def exposure_disclosure(chokepoint: str = "CP_HORMUZ") -> pd.DataFrame:
    """Per-country import dependence iota_n, route share theta_n, and exposure delta_n = min(1, iota_n * theta_n), so national results can be reconstructed. Reads calibration data and IMPORT_SHARE; no solve.
    """
    net = get_network()
    attr = _DEP_ATTR_BY_CP.get(chokepoint, "hormuz_dep")
    rows = []
    for code, c in net.countries.items():
        theta = float(getattr(c, attr, 0.0))
        iota = float(IMPORT_SHARE.get(code, 1.0))
        delta = min(1.0, iota * theta)
        rows.append({"Country": c.name, "Code": code,
                     "Import dependence iota_n": round(iota, 2),
                     "Route share theta_n": round(theta, 2),
                     "Exposure delta_n": round(delta, 3),
                     "Demand (mbd)": round(c.daily_oil_mbd, 3)})
    return pd.DataFrame(rows).sort_values("Exposure delta_n", ascending=False)\
        .reset_index(drop=True)


def normal_state_reconciliation(chokepoint: str = "CP_HORMUZ") -> pd.DataFrame:
    """Normal-state material balance: total importer demand split into always-available non-exposed supply and chokepoint-exposed imports (mutually exclusive, summing to demand), with global source capacity, passage throughput, and off-strait spare capacity as memos. Rates in mbd; no solve.
    """
    net = get_network()
    attr = _DEP_ATTR_BY_CP.get(chokepoint, "hormuz_dep")
    tot_demand = sum(c.daily_oil_mbd for c in net.countries.values())
    exposed = sum(min(1.0, IMPORT_SHARE.get(code, 1.0) * float(getattr(c, attr, 0.0)))
                  * c.daily_oil_mbd for code, c in net.countries.items())
    unaffected = tot_demand - exposed
    src_cap = sum(a.capacity_mbd for a in net.arcs
                  if a.origin.startswith("SRC_") and not a.is_bypass)
    through = net.chokepoints[chokepoint].throughput_mbd
    rows = [
        {"Component": "Total importer demand", "mbd": round(tot_demand, 1),
         "Note": "sum of the 53 importers"},
        {"Component": "  Always-available (non-exposed) supply", "mbd": round(unaffected, 1),
         "Note": "domestic and non-routed imports, (1 - delta_n) share"},
        {"Component": "  Chokepoint-exposed imports", "mbd": round(exposed, 1),
         "Note": "delta_n share, at risk from the closure"},
        {"Component": "Memo: global source capacity", "mbd": round(src_cap, 1),
         "Note": "network source injections"},
        {"Component": f"Memo: {chokepoint} normal throughput", "mbd": round(through, 1),
         "Note": "EIA calibrated passage volume"},
        {"Component": "Memo: off-strait spare capacity", "mbd": round(GLOBAL_SPARE_CAPACITY_MBD, 1),
         "Note": "reaches market without crossing the passage"},
    ]
    return pd.DataFrame(rows)


# FIGURES: matplotlib/cartopy publication charts driven by the result tables, plus the standalone chokepoint map.

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath




# Conventional national centroids (lon, lat) for the 53 importers, only to place map bubbles.
COUNTRY_LONLAT: Dict[str, tuple] = {
    "JPN": (138.2, 36.2), "KOR": (127.8, 36.5), "IND": (79.0, 22.0),
    "CHN": (104.0, 35.0), "SGP": (103.8, 1.35), "THA": (101.0, 15.0),
    "TWN": (121.0, 23.7), "PHL": (122.0, 12.0), "PAK": (69.3, 30.0),
    "BGD": (90.3, 23.7), "LKA": (80.7, 7.9), "VNM": (106.0, 16.0),
    "MYS": (102.0, 4.0), "IDN": (113.0, -2.5), "MMR": (96.0, 21.0),
    "KHM": (105.0, 12.5), "NPL": (84.1, 28.2), "HKG": (114.1, 22.3),
    "NZL": (172.0, -41.0), "AUS": (134.0, -25.0), "DEU": (10.4, 51.2),
    "ITA": (12.6, 42.5), "ESP": (-3.7, 40.2), "FRA": (2.3, 46.6),
    "GRC": (22.0, 39.2), "NLD": (5.3, 52.2), "BEL": (4.5, 50.6),
    "POL": (19.1, 52.1), "PRT": (-8.2, 39.6), "SWE": (15.6, 62.0),
    "GBR": (-1.5, 52.6), "AUT": (14.1, 47.6), "CZE": (15.5, 49.8),
    "ROU": (25.0, 45.9), "TUR": (35.2, 39.0), "EGY": (30.0, 26.8),
    "ISR": (34.9, 31.4), "JOR": (36.2, 31.2), "LBN": (35.9, 33.9),
    "MAR": (-6.8, 31.8), "ETH": (39.6, 9.1), "KEN": (37.9, 0.2),
    "UGA": (32.3, 1.4), "DJI": (42.6, 11.6), "RWA": (29.9, -1.9),
    "ZAF": (24.7, -29.0), "NGA": (8.1, 9.6), "TZA": (34.9, -6.4),
    "MOZ": (35.5, -17.3), "BRA": (-51.9, -10.8), "CHL": (-71.0, -35.7),
    "ARG": (-65.2, -35.4), "COL": (-74.3, 4.6),
}

REGION_COLOR = {
    "Asia-Pac.": "#2C6FB2", "Europe": "#2E8B57", "MENA": "#C8860C",
    "Africa": "#9C4DA0", "Americas": "#C0504D", "Oceania": "#4E8C8A",
}


# Data export (read straight from the solved MIP)
def export_country_losses(output_dir: str = "outputs",
                          target_count: int = 15) -> pd.DataFrame:
    """All-53 per-country inaction vs proactive expected loss with coordinates."""
    countries = build_countries()
    m0 = mt.solve((), target_count=target_count, cache=False)
    d0 = mt.country_damage(m0)
    del m0
    m1 = mt.solve(tuple(ALL_POLICIES), target_count=target_count, cache=False)
    d1 = mt.country_damage(m1)
    del m1
    rows = []
    for code, c in countries.items():
        inaction = float(d0.get(code, 0.0))
        proactive = float(d1.get(code, 0.0))
        lon, lat = COUNTRY_LONLAT.get(code, (np.nan, np.nan))
        rows.append({
            "Code": code, "Country": c.name, "Region": c.region,
            "Inaction ($B)": round(inaction, 2),
            "Proactive ($B)": round(proactive, 2),
            "Saving ($B)": round(inaction - proactive, 2),
            "Saving (%)": round(100.0 * (inaction - proactive) / inaction, 1)
            if inaction > 1e-9 else 0.0,
            "lon": lon, "lat": lat,
        })
    df = pd.DataFrame(rows).sort_values("Inaction ($B)", ascending=False)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "data_country_losses.csv"), index=False)
    return df.reset_index(drop=True)


def _node_layer(node_id: str) -> Optional[str]:
    """Classify a network node into a Sankey layer by its id prefix."""
    if node_id.startswith("SRC_"):
        return "source"
    if node_id.startswith("CP_"):
        return "chokepoint"
    if node_id.startswith("PROC_"):
        return "processing"
    return "importer"  # bare country code


def export_optimized_flows(output_dir: str = "outputs",
                           policy_subset=tuple(ALL_POLICIES),
                           target_count: int = 15) -> pd.DataFrame:
    """Expected optimized arc flow (mbd), aggregated by echelon transition.

    Link list for a layered Sankey (source regions -> chokepoints/bypass -> processing hubs -> importing regions); each arc flow is the prob-weighted, stage-averaged optimized flow.
    """
    import pyomo.environ as pyo
    net = get_network()
    arcs = {a.aid: a for a in net.arcs}
    countries = build_countries()
    m = mt.solve(tuple(policy_subset), target_count=target_count, cache=False)
    n_stages = len(m.T)
    pw = realized_scenario_probs(m)

    # x is a per-stage volume (mbd x stage-days); convert to an average rate in mbd by prob-weighting, summing stage volumes, and dividing by the horizon (n_stages x STAGE_DAYS).
    horizon_days = n_stages * STAGE_DAYS
    arc_flow = {}
    for aid in m.A:
        vol = sum(pw[w] * pyo.value(m.x[aid, t, w]) for t in m.T for w in m.OMEGA)
        arc_flow[aid] = vol / horizon_days
    del m

    def label(node_id, layer):
        if layer == "importer":
            c = countries.get(node_id)
            return c.region if c else node_id
        return node_id.replace("SRC_", "").replace("CP_", "").replace("PROC_", "")

    links = {}
    for aid, flow in arc_flow.items():
        if flow <= 1e-6:
            continue
        a = arcs[aid]
        lo, ld = _node_layer(a.origin), _node_layer(a.destination)
        src = (lo, label(a.origin, lo))
        dst = (ld, label(a.destination, ld))
        kind = "bypass" if a.is_bypass else ("sharing" if a.is_sharing else "main")
        links[(src, dst, kind)] = links.get((src, dst, kind), 0.0) + flow

    rows = [{
        "src_layer": s[0], "src": s[1], "dst_layer": d[0], "dst": d[1],
        "kind": k, "flow_mbd": round(v, 3),
    } for (s, d, k), v in sorted(links.items(), key=lambda kv: -kv[1])]
    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "data_optimized_flows.csv"), index=False)
    return df


# Figure 1: country loss map
def _iso3(att):
    """Robust ISO3 code from a Natural Earth country record's attributes."""
    for key in ("ISO_A3", "ISO_A3_EH", "ADM0_A3", "SOV_A3"):
        v = att.get(key)
        if v and v not in ("-99", "", None):
            return v
    return None


def make_country_loss_map(df: Optional[pd.DataFrame] = None,
                          output_dir: str = "outputs",
                          metric: str = "Inaction ($B)") -> str:
    """Choropleth world map of national expected economic loss under inaction.

    Importers filled by loss on a light basemap, styled after recent maritime-chokepoint impact maps (Verschuur et al., 2025). Cartopy + matplotlib (Natural Earth admin_0 polygons keyed on ISO-3), vector PDF; falls back to a plain scatter if Cartopy is unavailable.
    """
    import matplotlib.colors as mcolors
    if df is None:
        df = export_country_losses(output_dir)
    d = df[df[metric] > 0].copy()
    vmax = float(d[metric].max())
    loss = dict(zip(d["Code"], d[metric]))

    try:
        import cartopy.crs as ccrs
        import cartopy.io.shapereader as shpreader
    except Exception:
        return _loss_map_fallback(d, metric, output_dir)

    # PowerNorm (gamma<1) keeps mid-sized importers distinguishable from the few very large ones.
    cmap = plt.cm.YlOrRd
    norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=vmax)
    OCEAN, NODATA, BORDER = "#EAF1F6", "#E8E8E8", "#FFFFFF"

    # Frame the region that actually bears the loss (plus the chokepoint).
    lons = [COUNTRY_LONLAT[c][0] for c in d["Code"] if c in COUNTRY_LONLAT] + [56.5]
    lats = [COUNTRY_LONLAT[c][1] for c in d["Code"] if c in COUNTRY_LONLAT] + [26.8]
    extent = [min(lons) - 13, max(lons) + 13, max(-58, min(lats) - 12),
              min(84, max(lats) + 14)]

    fig = plt.figure(figsize=(12.4, 6.8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)

    shp = shpreader.natural_earth(resolution="110m", category="cultural",
                                  name="admin_0_countries")
    for rec in shpreader.Reader(shp).records():
        code = _iso3(rec.attributes)
        face = cmap(norm(loss[code])) if code in loss else NODATA
        ax.add_geometries([rec.geometry], ccrs.PlateCarree(), facecolor=face,
                          edgecolor=BORDER, linewidth=0.35,
                          zorder=2 if code in loss else 1)
    ax.axis("off")

    # disrupted chokepoint marker
    ax.plot(56.5, 26.8, marker="o", markersize=8, color="#15314C",
            markeredgecolor="white", markeredgewidth=1.0, zorder=6,
            transform=ccrs.PlateCarree())
    ax.annotate("Strait of Hormuz", (56.5, 26.8), xytext=(-9, -11),
                textcoords="offset points", fontsize=9, color="#15314C", zorder=7,
                ha="right", va="top",
                path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])

    # Label larger importers, greedily skipping overlaps so dense clusters stay legible; color for contrast against the fill.
    placed = []
    for _, r in d.sort_values(metric, ascending=False).iterrows():
        if r["Code"] not in COUNTRY_LONLAT or len(placed) >= 10:
            continue
        lo, la = COUNTRY_LONLAT[r["Code"]]
        if any(abs(lo - p[0]) < 16 and abs(la - p[1]) < 8 for p in placed):
            continue
        placed.append((lo, la))
        col = "white" if r[metric] > 0.55 * vmax else "#1A1A1A"
        ax.text(lo, la, r["Country"], fontsize=8.5, ha="center", va="center",
                color=col, zorder=8, transform=ccrs.PlateCarree(),
                path_effects=[pe.withStroke(
                    linewidth=1.8, foreground="#333333" if col == "white" else "white")])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Tick at the true maximum, since round(vmax) could exceed vmax and be clipped off the bar.
    small = [t for t in (1, 3, 5, 10, 20, 40, 70) if t < 0.9 * vmax]
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                      pad=0.02, aspect=42, ticks=small + [vmax])
    cb.set_ticklabels([str(t) for t in small] + [str(int(round(vmax)))])
    cb.set_label("Expected economic loss under inaction (USD billion)", fontsize=11)
    cb.ax.tick_params(labelsize=9, length=2)
    cb.outline.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.06)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fig_country_loss_map.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close()
    return path


def _loss_map_fallback(d, metric, output_dir):
    """Minimal scatter fallback if Cartopy is unavailable."""
    import matplotlib.colors as mcolors
    norm = mcolors.PowerNorm(gamma=0.45, vmin=0.0, vmax=float(d[metric].max()))
    fig, ax = plt.subplots(figsize=(12.6, 6.6))
    ax.set_facecolor("#EAF1F6")
    ax.set_xlim(-108, 162)
    ax.set_ylim(-56, 80)
    sc = ax.scatter(d["lon"], d["lat"], s=70, c=d[metric], cmap=plt.cm.YlOrRd,
                    norm=norm, edgecolors="#333333", linewidths=0.5)
    fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.04,
                 label=f"Expected loss ({metric})")
    ax.set_xticks([])
    ax.set_yticks([])
    path = os.path.join(output_dir, "fig_country_loss_map.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


# Figure 2: optimized-flow Sankey (pure matplotlib)
def _ribbon(ax, x0, y0a, y0b, x1, y1a, y1b, color, alpha=0.55):
    """Draw a smooth flow ribbon between two vertical spans."""
    xm = (x0 + x1) / 2.0
    verts = [
        (x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a),
        (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color,
                                    edgecolor="none", alpha=alpha, zorder=2))


# Readable display names for the terse network node codes.
_NODE_NAME = {
    "GULF": "Persian Gulf", "SEASI": "SE Asia", "NAFR": "N. Africa",
    "NAFTA": "N. America", "CASP": "Caspian", "LATAM": "Lat. America",
    "NSEA": "North Sea", "RUSIA": "Russia", "AUSNZ": "Aus/NZ", "WAFR": "W. Africa",
    "HORMUZ": "Hormuz", "MALACCA": "Malacca", "SUEZ": "Suez",
    "BABEL": "Bab-el-Mandeb",
    "JAMNAGAR": "Jamnagar", "FUJAIRAH": "Fujairah", "TANJUNG": "Tanjung",
    "MUMBAI": "Mumbai", "JURONG": "Jurong", "ROTTER": "Rotterdam",
    "SIDI": "Sidi Kerir", "YANBU": "Yanbu", "QINGDAO": "Qingdao",
    "ULSAN": "Ulsan", "HOUSTON": "Houston", "DJIBOUTI": "Djibouti",
    "OTHERSRC": "Other sources", "OTHERPROC": "Other refineries",
}


def _disp(code):
    return _NODE_NAME.get(code, code.title() if str(code).isupper() else str(code))


def make_flow_sankey(df: Optional[pd.DataFrame] = None,
                     output_dir: str = "outputs") -> str:
    """Matplotlib Sankey of optimized crude oil flows under the proactive response: source regions -> chokepoints -> refining hubs -> importing regions.

    Layer-colored node bars, ribbons colored by route type, labels under their column header (sources left, importers right, middle columns centered). Vector PDF, no browser engine.
    """
    if df is None:
        df = export_optimized_flows(output_dir)

    LAYERS = ["source", "chokepoint", "processing", "importer"]
    LX = {"source": 0.0, "chokepoint": 1.7, "processing": 3.4, "importer": 5.1}
    LAYER_COLOR = {"source": "#4E79A7", "chokepoint": "#B0742B",
                   "processing": "#59A14F", "importer": "#7E6BA8"}
    KIND_COLOR = {"main": "#5B86B0", "bypass": "#E0A458", "sharing": "#6FB1A6"}

    # Collapse negligible nodes in a layer into one "Other ..." node so each column is a short readable stack; dominant flows are left intact.
    df = df.copy()
    for layer, thresh, newcode in (("source", 0.25, "OTHERSRC"),
                                   ("processing", 0.50, "OTHERPROC")):
        out_s = df[df["src_layer"] == layer].groupby("src")["flow_mbd"].sum()
        in_s = df[df["dst_layer"] == layer].groupby("dst")["flow_mbd"].sum()
        thru = {n: max(float(out_s.get(n, 0.0)), float(in_s.get(n, 0.0)))
                for n in set(out_s.index) | set(in_s.index)}
        minor = {n for n, v in thru.items() if v < thresh}
        if len(minor) >= 2:
            df.loc[(df["src_layer"] == layer) & (df["src"].isin(minor)),
                   "src"] = newcode
            df.loc[(df["dst_layer"] == layer) & (df["dst"].isin(minor)),
                   "dst"] = newcode
            df = df[~((df["src_layer"] == df["dst_layer"]) &
                      (df["src"] == df["dst"]))]  # drop any self-loops
            df = df.groupby(["src_layer", "src", "dst_layer", "dst", "kind"],
                            as_index=False)["flow_mbd"].sum()

    in_tot, out_tot = {}, {}
    for _, r in df.iterrows():
        out_tot[(r["src_layer"], r["src"])] = \
            out_tot.get((r["src_layer"], r["src"]), 0.0) + r["flow_mbd"]
        in_tot[(r["dst_layer"], r["dst"])] = \
            in_tot.get((r["dst_layer"], r["dst"]), 0.0) + r["flow_mbd"]
    nodes = set(in_tot) | set(out_tot)
    size = {n: max(in_tot.get(n, 0.0), out_tot.get(n, 0.0)) for n in nodes}

    by_layer = {l: [] for l in LAYERS}
    for n in nodes:
        by_layer[n[0]].append(n)
    for l in LAYERS:
        by_layer[l].sort(key=lambda n: -size[n])

    # Vertical layout: each node's slot height is floored so tiny nodes still fit a label, but the bar is drawn at the true flow height so ribbon widths stay proportional.
    GAP = 0.55
    slot, center = {}, {}
    for l in LAYERS:
        ns = by_layer[l]
        big = max(size[n] for n in ns)
        for n in ns:
            slot[n] = max(size[n], 0.16 * big)
        total = sum(slot[n] for n in ns) + GAP * max(0, len(ns) - 1)
        y = total / 2.0
        for n in ns:
            center[n] = y - slot[n] / 2.0
            y -= slot[n] + GAP

    fig, ax = plt.subplots(figsize=(13.0, 8.2))
    NW = 0.085  # node bar half-width
    # stack ribbons from the top of each node's bar (separate in/out edges)
    out_off = {n: center[n] + size[n] / 2.0 for n in nodes}
    in_off = {n: center[n] + size[n] / 2.0 for n in nodes}
    for _, r in df.sort_values("flow_mbd", ascending=False).iterrows():
        s = (r["src_layer"], r["src"])
        d = (r["dst_layer"], r["dst"])
        h = r["flow_mbd"]
        x0, x1 = LX[s[0]] + NW, LX[d[0]] - NW
        _ribbon(ax, x0, out_off[s], out_off[s] - h,
                x1, in_off[d], in_off[d] - h,
                KIND_COLOR.get(r["kind"], "#5B86B0"), alpha=0.45)
        out_off[s] -= h
        in_off[d] -= h

    for n in nodes:
        x, c, hh = LX[n[0]], center[n], size[n]
        ax.add_patch(mpatches.Rectangle((x - NW, c - hh / 2.0), 2 * NW, hh,
                                        facecolor=LAYER_COLOR[n[0]],
                                        edgecolor="white", linewidth=0.5,
                                        zorder=4))
        name = _disp(n[1])
        if n[0] == "source":
            ax.text(x - NW - 0.12, c, name, ha="right", va="center",
                    fontsize=9.5, color="#1A1A1A", zorder=6)
        elif n[0] == "importer":
            ax.text(x + NW + 0.12, c, name, ha="left", va="center",
                    fontsize=9.5, color="#1A1A1A", zorder=6)
        else:
            ax.text(x, c, name, ha="center", va="center", fontsize=9,
                    color="#1A1A1A", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                              alpha=0.74, edgecolor="none"))

    ymax = max(center[n] + slot[n] / 2.0 for n in nodes)
    ymin = min(center[n] - slot[n] / 2.0 for n in nodes)
    heads = {"source": "Source region", "chokepoint": "Chokepoint",
             "processing": "Refining hub", "importer": "Importing region"}
    for l in LAYERS:
        ax.text(LX[l], ymax + 0.9, heads[l], ha="center", va="bottom",
                fontsize=12, color="#333333")
    ax.set_title("Optimized crude oil flows under the proactive response",
                 fontsize=14, color="#1A1A1A", pad=24)

    handles = [mpatches.Patch(facecolor=c, edgecolor="none", alpha=0.8,
                              label={"main": "Primary sea route",
                                     "bypass": "Bypass / pipeline",
                                     "sharing": "Bilateral sharing"}[k])
               for k, c in KIND_COLOR.items() if k in set(df["kind"])]
    ax.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
              bbox_to_anchor=(0.5, -0.05), fontsize=10)
    ax.text(LX["importer"] + 1.05, ymax + 0.9, "(mbd)", ha="left", va="bottom",
            fontsize=9, color="#8A94A0")

    ax.set_xlim(-1.55, LX["importer"] + 1.55)
    ax.set_ylim(ymin - 0.9, ymax + 1.9)
    ax.axis("off")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fig_flow_sankey.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close()
    return path


# EIA-style palette
GOLD = "#F2A900"        # chokepoint bubble fill
GOLD_EDGE = "#C8860C"   # bubble outline
ARROW = "#8C2E2E"       # crude oil flow arrows (dark red)
LAND = "#D9D9D9"        # land fill (light grey)
OCEAN = "#E3EDF4"       # ocean (pale blue), drawn as axes background
NUMCOL = "#23303F"      # number printed inside the bubble
LABELCOL = "#2B2B2B"    # chokepoint name labels


def _smooth(pts, n=200):
    """Return a smooth Catmull-Rom curve (lon, lat arrays) through waypoints."""
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        t = np.linspace(0, 1, n)
        return (pts[0, 0] + (pts[-1, 0] - pts[0, 0]) * t,
                pts[0, 1] + (pts[-1, 1] - pts[0, 1]) * t)
    p = np.vstack([pts[0], pts, pts[-1]])
    xs, ys = [], []
    seg = max(2, n // (len(pts) - 1))
    for i in range(1, len(p) - 2):
        P0, P1, P2, P3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, seg, endpoint=False):
            t2, t3 = t * t, t * t * t
            a = 0.5 * (2 * P1 + (-P0 + P2) * t
                       + (2 * P0 - 5 * P1 + 4 * P2 - P3) * t2
                       + (-P0 + 3 * P1 - 3 * P2 + P3) * t3)
            xs.append(a[0])
            ys.append(a[1])
    xs.append(pts[-1, 0])
    ys.append(pts[-1, 1])
    return np.array(xs), np.array(ys)


def generate_chokepoint_map(output_dir: str = "outputs") -> str:
    """Generate the EIA-style chokepoint map and save it as PDF (+ PNG)."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(13.2, 7.2))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-140, 152, -55, 72], crs=ccrs.PlateCarree())
    ax.set_facecolor(OCEAN)

    ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor=LAND,
                   edgecolor="none", zorder=1)
    for feat, lw in ((cfeature.COASTLINE, 0.4), (cfeature.BORDERS, 0.5)):
        try:
            ax.add_feature(feat.with_scale("110m"), edgecolor="white",
                           linewidth=lw, zorder=2)
        except Exception:
            pass
    ax.axis("off")

    # Principal crude oil sea lanes (waypoints kept over open water).
    routes = [
        ([(56.5, 26.8), (59.5, 24), (63, 16), (72, 7.5), (82, 5),
          (92, 4.5), (100.5, 2.8)], 2.6),
        ([(100.5, 2.8), (106, 7), (112, 13), (116, 19), (119, 25),
          (121, 29)], 2.4),
        ([(100.5, 2.8), (107, 8), (114, 15), (119, 21), (124, 27),
          (130, 32)], 2.2),
        ([(56.5, 26.8), (59.5, 23), (61, 15), (55, 13), (49, 12.5),
          (43.6, 12.7)], 2.4),
        ([(43.4, 12.6), (41, 16), (38.5, 20), (35.5, 24.5), (33.5, 27.5),
          (32.6, 29.8)], 2.2),
        ([(32.5, 30.4), (31, 32), (27, 33.8), (18, 35.4), (10, 37),
          (2, 37.5), (-3, 36.5), (-6, 36)], 2.2),
        ([(61, 13), (58, 2), (56, -12), (52, -22), (44, -30), (32, -34.5),
          (22, -35)], 2.4),
        ([(20, -34.8), (11, -31), (8, -18), (4, -5), (0, 3), (-11, 7),
          (-19, 12), (-24, 20), (-26, 29), (-16, 35), (-9, 37)], 2.2),
        ([(20, -34.8), (8, -34), (-12, -31), (-32, -27), (-46, -25)], 2.0),
        ([(-80.5, 8.5), (-88, 11), (-98, 15), (-110, 21), (-122, 31)], 2.0),
        ([(-78.5, 9.5), (-74, 14), (-72, 20), (-74, 27), (-77, 32)], 1.9),
        ([(17, 56), (13, 56.2), (11, 56.5), (8, 57.3), (3, 58)], 1.9),
        ([(36, 44), (32, 43), (29, 41), (26, 39), (24, 36.5)], 1.8),
    ]
    for wp, w in routes:
        xs, ys = _smooth(wp)
        ax.plot(xs, ys, color=ARROW, lw=w, alpha=0.9, solid_capstyle="round",
                solid_joinstyle="round", zorder=4, transform=ccrs.PlateCarree())
        ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-7], ys[-7]),
                    transform=ccrs.PlateCarree(),
                    arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=w,
                                    mutation_scale=16), zorder=5)

    # Chokepoint bubbles: (lon, lat, throughput mbd, dx, dy, ha, va).
    chokepoints = {
        "Strait of Hormuz":  (56.5, 26.8, 20.9,   0,  32, "center", "bottom"),
        "Strait of Malacca": (100.5, 3.0, 23.2,   0, -24, "center", "top"),
        "Suez Canal":        (32.5, 30.2,  4.9, -22,   8, "right",  "bottom"),
        "Bab el-Mandeb":     (43.4, 12.6,  4.2,  15,  -9, "left",   "top"),
        "Cape of Good Hope": (20.0, -34.8, 9.1,   0, -18, "center", "top"),
        "Panama Canal":      (-79.5, 9.0,  2.3, -16,  -6, "right",  "top"),
        "Danish Straits":    (11.0, 56.0,  4.9,   0,  15, "center", "bottom"),
        "Turkish Straits":   (29.0, 41.0,  3.7,  10,  10, "left",   "bottom"),
    }
    for name, (lon, lat, flow, dx, dy, ha, va) in chokepoints.items():
        size = flow * 150 + 260
        ax.scatter(lon, lat, s=size, c=GOLD, edgecolors=GOLD_EDGE,
                   linewidth=1.2, zorder=8, transform=ccrs.PlateCarree())
        fs = min(max(flow * 0.42 + 9.0, 13.0), 15.0)
        ax.text(lon, lat, f"{flow:.1f}", ha="center", va="center",
                fontsize=fs, fontweight="bold", color=NUMCOL, zorder=9,
                transform=ccrs.PlateCarree())
        ax.annotate(name, (lon, lat), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=13, fontweight="bold", color=LABELCOL, zorder=10,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    savepath = os.path.join(output_dir, "fig_chokepoint_map.pdf")
    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.savefig(savepath.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close()
    return savepath


if __name__ == "__main__":
    generate_chokepoint_map()


# VALIDATION: correctness invariants for the multi-stage stochastic model.
#
# Each check re-derives a property from a fresh solve and reports PASS/FAIL; run
# `python code/analysis.py` to verify after any formulation change. Invariants:
#   1. DDU linearization exact: at de-escalation strength zero the endogenous
#      objective equals the exogenous solve.
#   2. Realized scenario probabilities sum to one.
#   3. Rerouted escape flow never exceeds the disrupted chokepoint's bypass capacity.
#   4. A full closure yields a strictly positive oil-price cost (channel active).
#   5. The objective decomposition reconciles to the solved objective.

import gc

import pyomo.environ as pyo


_TOL = 1e-3


def _solve(policy_subset, scenario_class, endogenous=True):
    net = get_network()
    sc = get_scenarios(scenario_class, 30)
    m = build_model(net, sc, disrupted_chokepoint=scen.disrupted_chokepoints_for_class(
                        scenario_class, "CP_HORMUZ"),
                    endogenous_uncertainty=endogenous,
                    unresponsive_regime=class_regime(scenario_class)[1],
                    policy_subset=list(policy_subset))
    info = solve_model(m, solver_name=_detect_solver(None), time_limit=600, mip_gap=1e-3)
    m._objective_value = info["objective"]
    return m


def check_ddu_exact():
    """rho=0 endogenous objective == exogenous objective (linearization exact)."""
    saved = scen.DEESCALATION_FACTOR
    scen.DEESCALATION_FACTOR = 0.0
    try:
        a = _solve(ALL_POLICIES, "full_closure", endogenous=False)._objective_value
        b = _solve(ALL_POLICIES, "full_closure", endogenous=True)._objective_value
    finally:
        scen.DEESCALATION_FACTOR = saved
    rel = abs(a - b) / max(1.0, abs(a))
    return rel <= _TOL, f"exogenous={a/1e3:.2f}B  DDU(rho=0)={b/1e3:.2f}B  rel={rel:.2e}"


def check_probs_normalize():
    """Realized scenario probabilities sum to one for inaction and full action."""
    msgs, ok = [], True
    for subset, lab in [((), "inaction"), (ALL_POLICIES, "full")]:
        m = _solve(subset, "full_closure")
        s = sum(realized_scenario_probs(m).values())
        ok &= abs(s - 1.0) <= 1e-6
        msgs.append(f"{lab}: sum={s:.6f}")
    return ok, "; ".join(msgs)


def check_escape_cap():
    """Rerouted escape flow per stage stays within the physical bypass capacity."""
    net = get_network()
    msgs, ok = [], True
    for cls in ["full_closure", "multi_chokepoint"]:
        m = _solve(ALL_POLICIES, cls)
        disrupted = scen.disrupted_chokepoints_for_class(cls, "CP_HORMUZ")
        disrupted = {disrupted} if isinstance(disrupted, str) else set(disrupted)
        for cp in disrupted:
            arcs = [a.aid for a in net.arcs if getattr(a, "relieves", "") == cp]
            if not arcs:
                continue
            cap = net.chokepoints[cp].bypass_capacity_mbd * float(STAGE_DAYS)
            mx = max(sum(pyo.value(m.x[a, t, w]) for a in arcs)
                     for t in m.T for w in m.OMEGA)
            ok &= mx <= cap + 1e-3
            msgs.append(f"{cls}/{cp}: max={mx:.1f} cap={cap:.1f}")
        del m
        gc.collect()
    return ok, "; ".join(msgs)


def check_price_floor():
    """A full closure raises the world price, so the oil-price term is positive."""
    m = _solve(ALL_POLICIES, "full_closure")
    price = mt.cost_components(m)["Oil price"]
    return price > 0.0, f"oil-price cost={price:.2f}B"


def check_cost_reconciles():
    """The objective decomposition sums to the solved objective."""
    m = _solve(ALL_POLICIES, "full_closure")
    cc = mt.cost_components(m)
    rel = abs(cc["Total"] * 1e3 - m._objective_value) / max(1.0, m._objective_value)
    return rel <= _TOL, f"components={cc['Total']:.2f}B obj={m._objective_value/1e3:.2f}B rel={rel:.2e}"


def check_cmax_margin():
    """Per-scenario cost stays below the McCormick bound Cmax with margin (a tight Cmax could silently distort or infeasibly cap the recourse)."""
    m = _solve(ALL_POLICIES, "full_closure")
    if not getattr(m, "ddu_active", False):
        return True, "n/a (DDU inactive)"
    mx = max(pyo.value(m.Cw[w]) for w in m.OMEGA)
    ratio = mx / m.ddu_Cmax
    return ratio < 0.9, f"max Cw={mx:.0f}  Cmax={m.ddu_Cmax:.0f}  ratio={ratio:.2f}"


_CHECKS = [
    ("DDU linearization exact", check_ddu_exact),
    ("Probabilities normalize", check_probs_normalize),
    ("Escape flow within capacity", check_escape_cap),
    ("Oil-price floor positive", check_price_floor),
    ("Cost decomposition reconciles", check_cost_reconciles),
    ("Cmax bound has margin", check_cmax_margin),
]


def run_all():
    """Run every invariant; return True if all pass."""
    all_ok = True
    for name, fn in _CHECKS:
        try:
            ok, detail = fn()
        except Exception as e:  # a crash is a failure
            ok, detail = False, f"ERROR: {e}"
        all_ok &= ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        gc.collect()
    print("ALL PASS" if all_ok else "SOME CHECKS FAILED")
    return all_ok


# STANDALONE PUBLICATION FIGURES (read locked outputs/*.csv)
# Styling for the descriptive figures, applied only within an rc_context so it does not leak into the notebook's serif style.
_PUB_RC = {
    "font.family": "Arial", "font.size": 11,
    "axes.linewidth": 0.8, "axes.edgecolor": "#3A3A3A",
    "axes.labelcolor": "#1A1A1A", "text.color": "#1A1A1A",
    "xtick.color": "#3A3A3A", "ytick.color": "#3A3A3A",
    "axes.labelsize": 12, "savefig.dpi": 400, "figure.dpi": 100,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def make_exposure_figure(out_dir="outputs"):
    """Figure 4 (fig_exposure_map): Hormuz dependence vs SPR coverage bubble
    chart, built directly from the calibration inputs in data_countries.csv.
    No solver needed."""
    import os
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FixedFormatter
    import matplotlib.patheffects as pe
    try:
        from adjustText import adjust_text
        have_adjust = True
    except Exception:
        have_adjust = False

    csv = os.path.join(out_dir, "data_countries.csv")
    out = os.path.join(out_dir, "fig_exposure_map")
    region_colors = {
        "Asia-Pac.": "#C0524F", "Europe": "#2F4B7C", "MENA": "#E0A458",
        "Africa": "#4F9D69", "Americas": "#8172B3", "Oceania": "#4E9CA6",
    }

    def darken(hexcolor, f=0.72):
        h = hexcolor.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return (r * f / 255, g * f / 255, b * f / 255)

    df = pd.read_csv(csv)
    df["x"] = df["Hormuz dep."] * 100.0
    df["y"] = df["SPR (days)"].astype(float)
    df["d"] = df["Demand (mbd)"].astype(float)

    def s_of_d(d):
        return (4.0 + 9.0 * np.sqrt(d)) ** 2

    df["s"] = s_of_d(df["d"])
    df = df.sort_values("d", ascending=False)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(8.8, 5.8))
        ax.set_yscale("log")
        ymin, ymax = 1.7, 320.0
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(-2, 103)
        ax.add_patch(Rectangle((50, ymin), 53, 30 - ymin, facecolor="#A83C3C",
                               alpha=0.06, edgecolor="none", zorder=0))
        ax.axvline(50, color="#A83C3C", lw=0.7, ls=(0, (4, 4)), alpha=0.35, zorder=0)
        ax.axhline(30, color="#A83C3C", lw=0.7, ls=(0, (4, 4)), alpha=0.35, zorder=0)
        ax.text(101.5, 28.5, "High-risk quadrant\nhigh dependence, low reserves",
                ha="right", va="top", fontsize=9.5, fontstyle="italic",
                color="#A83C3C", alpha=0.9, linespacing=1.25)
        for _, r in df.iterrows():
            ax.scatter(r["x"], r["y"], s=r["s"],
                       c=[region_colors.get(r["Region"], "#7F8C99")],
                       alpha=0.82, edgecolors="white", linewidth=0.7, zorder=3)
        texts = []
        for _, r in df.iterrows():
            texts.append(ax.text(r["x"], r["y"], r["Code"], fontsize=7.6,
                         fontweight="bold", ha="center", va="center",
                         color=darken(region_colors.get(r["Region"], "#7F8C99")),
                         path_effects=[pe.withStroke(linewidth=1.7, foreground="white")],
                         zorder=5))
        if have_adjust:
            adjust_text(texts, ax=ax, expand=(1.18, 1.35), force_text=(0.5, 0.7),
                        ensure_inside_axes=True, max_move=22,
                        arrowprops=dict(arrowstyle="-", color="#9AA0A6", lw=0.5,
                                        shrinkA=4, shrinkB=2, alpha=0.8))
        ax.set_xlabel("Strait of Hormuz dependence (% of crude imports)")
        ax.set_ylabel("Strategic petroleum reserve coverage (days)")
        yt = [2, 5, 10, 20, 30, 50, 100, 200]
        ax.yaxis.set_major_locator(FixedLocator(yt))
        ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in yt]))
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(0, 101, 20))
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", which="major", color="#E3E6EA", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        region_order = ["Asia-Pac.", "Europe", "MENA", "Africa", "Americas", "Oceania"]
        reg_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=8,
                       markerfacecolor=region_colors[k], markeredgecolor="white",
                       markeredgewidth=0.6, label=k) for k in region_order]
        leg1 = ax.legend(handles=reg_handles, title="Region", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10,
                         title_fontsize=10.5, handletextpad=0.4, labelspacing=0.6)
        leg1.get_title().set_fontweight("bold")
        ax.add_artist(leg1)
        size_vals = [0.5, 2, 5, 15]
        size_handles = [Line2D([0], [0], marker="o", linestyle="",
                        markerfacecolor="#9AA0A6", markeredgecolor="white",
                        markeredgewidth=0.6, markersize=np.sqrt(s_of_d(v)),
                        label=f"{v:g} mbd") for v in size_vals]
        leg2 = ax.legend(handles=size_handles, title="Crude oil demand", loc="lower left",
                         bbox_to_anchor=(1.01, 0.0), frameon=False, fontsize=10,
                         title_fontsize=10.5, labelspacing=1.4, handletextpad=0.8,
                         borderpad=1.0)
        leg2.get_title().set_fontweight("bold")
        fig.subplots_adjust(left=0.085, right=0.79, top=0.97, bottom=0.105)
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png")
    return fig


def make_pareto_figure(out_dir="outputs"):
    """Figure 9 (fig_pareto_frontier2): efficiency-equity frontier by scenario
    severity, rebuilt from data_pareto_frontier_mip.csv. No solver needed."""
    import os
    import matplotlib.patheffects as pe
    from matplotlib.ticker import PercentFormatter

    csv = os.path.join(out_dir, "data_pareto_frontier_mip.csv")
    out = os.path.join(out_dir, "fig_pareto_frontier2")
    scen_spec = [
        ("partial_restriction", "Partial restriction", "#E0A458"),
        ("full_closure",        "Full closure",        "#DD8452"),
        ("multi_chokepoint",    "Multi-chokepoint",    "#A83C3C"),
    ]
    df = pd.read_csv(csv)

    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        ends = {}
        for key, label, color in scen_spec:
            s = df[df["Scenario"] == key].sort_values("Equity weight")
            loss = s["Aggregate loss ($B)"].to_numpy(float)
            gini = s["Oil-access Gini"].to_numpy(float)
            x = (loss / loss[0] - 1.0) * 100.0
            y = (1.0 - gini / gini[0]) * 100.0
            ax.plot(x, y, "-", color=color, lw=2.4, zorder=3, solid_capstyle="round")
            ax.scatter(x, y, s=26, color=color, edgecolors="white", linewidth=0.8,
                       zorder=4, label=label)
            ends[key] = (x[-1], y[-1], color)
        ax.axhline(0, color="#C8CDD2", lw=0.8, zorder=1)
        ax.axvline(0, color="#C8CDD2", lw=0.8, zorder=1)
        sm = df[df["Scenario"] == "multi_chokepoint"].sort_values("Equity weight")
        lm = sm["Aggregate loss ($B)"].to_numpy(float)
        gm = sm["Oil-access Gini"].to_numpy(float)
        xm = (lm / lm[0] - 1.0) * 100.0
        ym = (1.0 - gm / gm[0]) * 100.0
        ax.annotate("", xy=(xm[-2], ym[-2]), xytext=(xm[3], ym[3]),
                    arrowprops=dict(arrowstyle="-|>", color="#6B7280", lw=1.4,
                                    connectionstyle="arc3,rad=0.18"), zorder=2)
        ax.text(0.62, 4.0, "stronger equity\nweighting", fontsize=9.5, color="#4B5563",
                fontstyle="italic", ha="left", va="center", linespacing=1.2)
        xs = max(ends["full_closure"][0], ends["multi_chokepoint"][0])
        ys = max(ends["full_closure"][1], ends["multi_chokepoint"][1])
        ax.annotate("Severe disruptions:\n~9% lower inequality\nfor ~2% more cost",
                    xy=(xs, ys), xytext=(xs - 0.95, ys + 0.2), fontsize=9.5,
                    fontweight="bold", color="#A83C3C", ha="right", va="top",
                    linespacing=1.25,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        xp, yp, cp = ends["partial_restriction"]
        ax.annotate("Partial restriction:\n~3% for ~1.3% cost",
                    xy=(xp, yp), xytext=(xp + 0.12, yp - 1.4), fontsize=9.5,
                    fontweight="bold", color="#B07A2E", ha="left", va="top",
                    linespacing=1.25,
                    arrowprops=dict(arrowstyle="-", color="#B07A2E", lw=0.7, alpha=0.7),
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
        ax.set_xlabel("Increase in aggregate expected damage")
        ax.set_ylabel("Reduction in oil-access inequality (Gini)")
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=1))
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_xlim(-0.1, 2.45)
        ax.set_ylim(-1.2, 10.2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="both", color="#EAEDF0", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        leg = ax.legend(loc="upper left", frameon=False, fontsize=10,
                        title="Scenario severity", title_fontsize=10.5,
                        handletextpad=0.5, labelspacing=0.5)
        leg.get_title().set_fontweight("bold")
        fig.tight_layout()
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png")
    return fig


def make_scenario_bars_figure(df_regime, out_dir="outputs"):
    """Figure 8 (fig_scenario_bars_mip): per-scenario expected loss under no
    intervention, the reactive bundle, and the early-stage optimum, with saving
    labels. Driven by the in-memory regime-comparison table (columns Scenario,
    No intervention, Reactive bundle, Full proactive)."""
    import os
    import matplotlib.patheffects as pe

    out = os.path.join(out_dir, "fig_scenario_bars_mip")
    colors = {"none": "#C0524F", "react": "#E0A458", "early": "#4F9D69"}
    df = df_regime.copy()
    df["Scenario"] = (df["Scenario"].astype(str)
                      .str.replace("Partial restriction", "Partial\nrestriction")
                      .str.replace("Full closure", "Full\nclosure")
                      .str.replace("Multi-chokepoint", "Multi-\nchokepoint"))
    df["save"] = (1.0 - df["Full proactive"] / df["No intervention"]) * 100.0

    x = np.arange(len(df))
    w = 0.26
    series = [
        ("No intervention", df["No intervention"], colors["none"], -w),
        ("Reactive bundle (P1+P2+P3)", df["Reactive bundle"], colors["react"], 0.0),
        ("Early-stage optimum (P1+P3+P6)", df["Full proactive"], colors["early"], w),
    ]
    with plt.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(8.6, 5.0))
        for label, vals, color, off in series:
            bars = ax.bar(x + off, vals, width=w, label=label, color=color,
                          edgecolor="white", linewidth=0.7, zorder=3)
            for b in bars:
                ax.annotate(f"{b.get_height():.0f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, color="#1A1A1A")
        for xi, (full, save) in enumerate(zip(df["Full proactive"], df["save"])):
            ax.annotate(f"−{save:.1f}%", (xi + w, full), xytext=(0, 16),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color="#2F6B45",
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        ax.set_xticks(x)
        ax.set_xticklabels(df["Scenario"], fontsize=10.5)
        ax.set_ylabel("Expected system loss across 53 importers (\\$B)")
        ax.set_ylim(0, df["No intervention"].max() * 1.16)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#E3E6EA", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=1.2,
                  handletextpad=0.5, labelspacing=0.5)
        fig.tight_layout()
        fig.savefig(out + ".pdf")
        fig.savefig(out + ".png")
    return fig


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_all() else 1)


# SENSITIVITY AND ROBUSTNESS CHECKS.
#
# Each function perturbs one calibrated input, re-solves the headline 100-day
# Hormuz closure, and reports the effect on the objective and optimal policy set,
# restoring the baseline on exit. Depends only on model. Key mappings:
#   cascade_beta_robustness -> beta intensity (+/-20% around the 2.58 anchor)
#   policy_cost_robustness  -> activation costs (common 0.5-2.0 factor)

import gc
from copy import deepcopy

import numpy as np
import pandas as pd
import pyomo.environ as pyo

import model
from model import (build_model, solve_model, get_network, get_scenarios,
                        disrupted_chokepoints_for_class, class_regime,
                        responsive_matrix, realized_scenario_probs,
                        build_countries, SECTORS, generate_full_tree,
                        ALL_POLICIES, POLICIES, _detect_solver)


# Cascade weights: first-order vs full Leontief inverse (no solve)
def cascade_weight_gap():
    """First-order (single-pass) cascade weights vs the total-requirements weights of the full Leontief inverse (I - beta)^{-1}.

    The model propagates stress in one pass, u = beta * sigma. Reports per sector the first-order and full-inverse row-sum weights, their relative gap, and the spectral radius of beta, showing the single pass under-states indirect stress (conservative) while preserving sector ranking. Reads model.BETA; no solve.
    """
    S = list(SECTORS)
    n = len(S)
    B = np.zeros((n, n))
    for a, row in model.BETA.items():
        for s, v in row.items():
            B[S.index(a), S.index(s)] = v
    inv = np.linalg.inv(np.eye(n) - B)
    w_full = inv - np.eye(n)          # beta + beta^2 + ... (indirect weights)
    rs1, rsf = B.sum(1), w_full.sum(1)
    rel = (rsf - rs1) / rs1
    rows = [{"Sector": S[i],
             "First-order row sum": round(float(rs1[i]), 3),
             "Full-inverse row sum": round(float(rsf[i]), 3),
             "Relative gap (%)": round(float(rel[i] * 100.0), 1)}
            for i in range(n)]
    spectral = float(max(abs(np.linalg.eigvals(B))))
    print(f"  spectral radius of beta = {spectral:.3f}; "
          f"max relative row-sum gap = {rel.max()*100:.1f}%")
    df = pd.DataFrame(rows)
    df.attrs["spectral_radius"] = round(spectral, 3)
    df.attrs["max_relative_gap_pct"] = round(float(rel.max() * 100.0), 1)
    return df


# ESCVI ranking stability under weight perturbation (no solve)
def escvi_weight_stability(base_weights=(0.30, 0.30, 0.20, 0.20),
                           jitter=(-0.25, -0.15, 0.0, 0.15, 0.25)):
    """Stability of the ESCVI ranking to its four dimension weights.

    Each weight is scaled by every combination of the jitter grid (default +/-25%), renormalized, and the ranking recomputed. Reports min and mean Spearman correlation to baseline and top-10/top-12 retention. No solve.
    """
    from analysis import compute_escvi
    df = compute_escvi(build_countries())
    dims = ["Import Dep.", "Reserve Vuln.", "Alt. Gap", "Cascade Sev."]
    X = df[dims].to_numpy()
    codes = df["Code"].to_numpy()
    base_w = np.array(base_weights)

    def ranking(w):
        return codes[np.argsort(-(X @ w))]

    base_rank = ranking(base_w)
    base_pos = {c: i for i, c in enumerate(base_rank)}
    base_top10, base_top12 = set(base_rank[:10]), set(base_rank[:12])

    rhos, r10, r12 = [], [], []
    for a in jitter:
        for b in jitter:
            for c in jitter:
                for d in jitter:
                    w = base_w * np.array([1 + a, 1 + b, 1 + c, 1 + d])
                    w = w / w.sum()
                    r = ranking(w)
                    pos = {cc: i for i, cc in enumerate(r)}
                    d1 = np.array([base_pos[cc] for cc in codes])
                    d2 = np.array([pos[cc] for cc in codes])
                    rhos.append(np.corrcoef(d1, d2)[0, 1])
                    r10.append(len(set(r[:10]) & base_top10) / 10.0)
                    r12.append(len(set(r[:12]) & base_top12) / 12.0)
    rhos, r10, r12 = np.array(rhos), np.array(r10), np.array(r12)
    print(f"  {len(rhos)} perturbations: Spearman rho min={rhos.min():.3f} "
          f"mean={rhos.mean():.3f}; top-12 retention min={r12.min()*100:.0f}%")
    return pd.DataFrame([{
        "Perturbations": len(rhos),
        "Spearman rho (min)": round(float(rhos.min()), 3),
        "Spearman rho (mean)": round(float(rhos.mean()), 3),
        "Top-10 retention (min, %)": round(float(r10.min() * 100), 0),
        "Top-12 retention (min, %)": round(float(r12.min() * 100), 0),
    }])


# Shared helpers
def _hormuz_setup(scenario_class, chokepoint, target_count):
    """Network, scenarios, disrupted chokepoint, unresponsive regime, and solver for a Hormuz solve (invariant across perturbations)."""
    return (get_network(),
            get_scenarios(scenario_class, target_count),
            disrupted_chokepoints_for_class(scenario_class, chokepoint),
            class_regime(scenario_class)[1],
            _detect_solver(None))


def _active_set(m):
    """Policies that activate in any stage/scenario of a solved model."""
    return tuple(p for p in ALL_POLICIES
                 if any(int(round(pyo.value(m.z[p, t, w]))) == 1
                        for t in m.T for w in m.OMEGA))


# Logistics realism: derated delivery capacity
def logistics_derating_robustness(factors=(1.0, 0.9, 0.8, 0.7),
                                  chokepoint="CP_HORMUZ",
                                  scenario_class="full_closure",
                                  duration_days=100,
                                  target_count=15):
    """Re-solve the closure with transport and rerouting arc capacities derated by a common factor (proxy for route lags, vessel-cycle limits, port congestion).

    Source-injection arcs stay at nameplate (they set the disruption size, not delivery logistics). Reports no-intervention and optimized loss, saving, and active set per derating. Baseline restored on exit.
    """
    network = get_network()
    disrupted = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    unresp = class_regime(scenario_class)[1]
    solver = _detect_solver(None)
    scenarios = get_scenarios(scenario_class, target_count)
    # Derate delivery/rerouting legs, not the source injections that size the shock.
    movable = [a for a in network.arcs if not a.origin.startswith("SRC_")]
    base_cap = {id(a): a.capacity_mbd for a in movable}

    def _loss(subset):
        m = build_model(network, scenarios, disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(subset))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError("logistics-derating solve infeasible")
        loss = float(info["objective"]) / 1000.0
        active = _active_set(m) if subset else ()
        del m
        gc.collect()
        return loss, active

    rows = []
    try:
        for f in factors:
            for a in movable:
                a.capacity_mbd = base_cap[id(a)] * f
            no_int, _ = _loss([])
            opt, active = _loss(ALL_POLICIES)
            rows.append({
                "Capacity factor": f,
                "No intervention ($B)": round(no_int, 1),
                "Optimized ($B)": round(opt, 1),
                "Saving (%)": round(100.0 * (1.0 - opt / no_int), 1),
                "Active policies": "+".join(active),
            })
            print(f"  cap x{f:.2f}: no-int={no_int:6.1f}B  opt={opt:6.1f}B  "
                  f"saving={100.0*(1.0-opt/no_int):4.1f}%  active={'+'.join(active)}")
    finally:
        for a in movable:
            a.capacity_mbd = base_cap[id(a)]
    return pd.DataFrame(rows)


# Duration under a FIXED decision clock
def _reduce_tree(scenarios, max_paths):
    """Cap a scenario tree at max_paths representative paths (highest probability,
    renormalized), so the model stays within memory as the stage count grows."""
    if len(scenarios) <= max_paths:
        return scenarios
    top = sorted(scenarios, key=lambda s: s.prob, reverse=True)[:max_paths]
    tot = sum(s.prob for s in top)
    return [model.Scenario(omega=i, prob=s.prob / tot, states=s.states,
                                availability=s.availability)
            for i, s in enumerate(top)]


def duration_fixed_stage_robustness(stage_counts=(4, 5),
                                    chokepoint="CP_HORMUZ",
                                    scenario_class="full_closure",
                                    stage_days=25,
                                    max_paths=27):
    """Re-solve the closure with a FIXED 25-day stage and increasing stage count, holding decision/information frequency constant as the horizon lengthens.

    The headline experiment instead keeps four stages and stretches each, coarsening the clock at long horizons. Reports fixed-stage saving beside the four-stage coarsened saving at the same duration, so a stable saving shows the decline is buffer exhaustion, not decision granularity. Tree grows as 3^(T-1) (T=4,5,6 give 27, 81, 243 paths). NUM_STAGES and the budget vector are monkeypatched and restored on exit.
    """
    network = get_network()
    disrupted = disrupted_chokepoints_for_class(scenario_class, chokepoint)
    init_state, unresp = class_regime(scenario_class)
    solver = _detect_solver(None)
    base_num_stages = model.NUM_STAGES
    base_budgets = list(model.STAGE_BUDGETS)

    def _solve(scen, n_stages, sd, subset):
        # Exogenous dynamics, matching the headline duration table and keeping larger trees within memory (the endogenous McCormick variant would make the 81-path T=5 model exhaust RAM).
        m = build_model(network, scen, disrupted_chokepoint=disrupted,
                        stage_days=sd, enable_cascade=True,
                        endogenous_uncertainty=False, unresponsive_regime=unresp,
                        policy_subset=list(subset))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError(f"fixed-stage solve infeasible (T={n_stages})")
        loss = float(info["objective"]) / 1000.0
        active = _active_set(m) if subset else ()
        del m
        gc.collect()
        return loss, active

    rows = []
    try:
        for n in stage_counts:
            duration = n * stage_days
            try:
                # Fixed 25-day clock, n stages: patch NUM_STAGES and budget vector. The tree grows as 3^(T-1), so cap it at max_paths representative paths for higher T. Both clocks use the same cap so the comparison stays like-for-like.
                model.NUM_STAGES = n
                model.STAGE_BUDGETS = (base_budgets + [base_budgets[-1]] * n)[:n]
                scen_fix = _reduce_tree(generate_full_tree(
                    initial_state=init_state, regime=unresp, num_stages=n,
                    stage_days=stage_days), max_paths)
                f_none, _ = _solve(scen_fix, n, stage_days, [])
                f_opt, f_active = _solve(scen_fix, n, stage_days, ALL_POLICIES)
                model.NUM_STAGES = base_num_stages
                model.STAGE_BUDGETS = list(base_budgets)

                # Four-stage coarsened clock at the same total duration.
                coarse_sd = duration / base_num_stages
                scen_coarse = _reduce_tree(generate_full_tree(
                    initial_state=init_state, regime=unresp,
                    num_stages=base_num_stages, stage_days=coarse_sd), max_paths)
                c_none, _ = _solve(scen_coarse, base_num_stages, coarse_sd, [])
                c_opt, _ = _solve(scen_coarse, base_num_stages, coarse_sd, ALL_POLICIES)
            except Exception as exc:  # noqa: BLE001
                # Any failure (including Gurobi OOM) skips that point rather than aborting the sweep.
                model.NUM_STAGES = base_num_stages
                model.STAGE_BUDGETS = list(base_budgets)
                gc.collect()
                print(f"  {duration}d (T={n}): SKIPPED ({type(exc).__name__}: {exc})")
                continue

            f_save = 100.0 * (1.0 - f_opt / f_none)
            c_save = 100.0 * (1.0 - c_opt / c_none)
            rows.append({
                "Duration (days)": duration,
                "Fixed-stage T": n,
                "Fixed-stage saving (%)": round(f_save, 1),
                "Four-stage saving (%)": round(c_save, 1),
                "Saving gap (pp)": round(f_save - c_save, 1),
                "Fixed-stage active": "+".join(f_active),
            })
            print(f"  {duration}d: fixed(T={n})={f_save:4.1f}%  "
                  f"coarse(T=4)={c_save:4.1f}%  gap={f_save-c_save:+.1f}pp  "
                  f"active={'+'.join(f_active)}")
    finally:
        model.NUM_STAGES = base_num_stages
        model.STAGE_BUDGETS = list(base_budgets)
    return pd.DataFrame(rows)


# Cascade intensity: beta +/- 20%
def cascade_beta_robustness(scales=(0.8, 1.0, 1.2),
                            chokepoint="CP_HORMUZ",
                            duration_days=100,
                            scenario_class="full_closure",
                            target_count=15):
    """Re-solve the Hormuz closure with the inter-sector dependence matrix beta scaled by each factor.

    Reports perturbed intensity (sum of beta), no-intervention and optimized loss, saving, and active set per scale. Baseline beta restored on exit. The +/-20% band brackets the calibrated intensity (~2.06 to 3.10 around the 2.58 anchor).
    """
    base_beta = {s: dict(d) for s, d in model.BETA.items()}
    base_intensity = sum(v for d in base_beta.values() for v in d.values())
    network, scenarios, disrupted, unresp, solver = _hormuz_setup(
        scenario_class, chokepoint, target_count)

    def _loss(policy_subset):
        m = build_model(network, scenarios, disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(policy_subset))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError("cascade-robustness solve infeasible")
        return m, float(info["objective"]) / 1000.0

    rows = []
    try:
        for sc in scales:
            model.BETA = {s: {k: round(v * sc, 4) for k, v in d.items()}
                               for s, d in base_beta.items()}
            _, no_int = _loss([])
            m_opt, opt = _loss(ALL_POLICIES)
            active = _active_set(m_opt)
            rows.append({
                "beta scale": sc,
                "Intensity": round(base_intensity * sc, 3),
                "No intervention ($B)": round(no_int, 1),
                "Optimized ($B)": round(opt, 1),
                "Saving (%)": round(100.0 * (1.0 - opt / no_int), 1),
                "Active policies": "+".join(active),
            })
            print(f"  beta x{sc:.2f}: no-int={no_int:6.1f}B  opt={opt:6.1f}B  "
                  f"saving={100.0*(1.0-opt/no_int):4.1f}%  active={'+'.join(active)}")
            del m_opt
            gc.collect()
    finally:
        model.BETA = base_beta
    return pd.DataFrame(rows)


# Policy activation costs: common factor 0.5 - 2.0
def policy_cost_robustness(factors=(0.5, 0.75, 1.0, 1.5, 2.0),
                           chokepoint="CP_HORMUZ",
                           duration_days=100,
                           scenario_class="full_closure",
                           target_count=15):
    """Re-solve the optimized Hormuz closure with every activation cost scaled by a common factor.

    Reports per factor the bundle cost, optimized loss, active set, and which instruments enter (+) or leave (-) relative to factor 1.0. Baseline costs restored on exit. Per-stage budgets are fixed, so at high factors an instrument can drop because the budget binds, not only the objective.
    """
    factors = tuple(sorted(set(factors) | {1.0}))
    base_costs = {p.pid: p.cost for p in POLICIES}
    base_bundle = sum(base_costs.values())
    network, scenarios, disrupted, unresp, solver = _hormuz_setup(
        scenario_class, chokepoint, target_count)

    def _solve_opt():
        m = build_model(network, scenarios, disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(ALL_POLICIES))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError("policy-cost robustness solve infeasible")
        return m, float(info["objective"]) / 1000.0

    rows, baseline_active = [], set()
    try:
        for f in factors:
            for p in POLICIES:
                p.cost = base_costs[p.pid] * f
            m_opt, opt = _solve_opt()
            active = _active_set(m_opt)
            if f == 1.0:
                baseline_active = set(active)
            rows.append({"cost factor": f,
                         "Bundle cost ($B)": round(base_bundle * f, 1),
                         "Optimized ($B)": round(opt, 1),
                         "_active": set(active),
                         "Active policies": "+".join(active)})
            print(f"  cost x{f:.2f}: opt={opt:7.1f}B  active={'+'.join(active)}")
            del m_opt
            gc.collect()
    finally:
        for p in POLICIES:
            p.cost = base_costs[p.pid]

    for r in rows:
        a = r.pop("_active")
        changes = ([f"+{x}" for x in sorted(a - baseline_active)]
                   + [f"-{x}" for x in sorted(baseline_active - a)])
        r["Change vs 1.0x"] = ", ".join(changes) if changes else "none"
    return pd.DataFrame(rows)


# Transition-matrix rows: adverse mass +/- delta
def transition_perturbation_robustness(deltas=(-0.10, -0.05, 0.0, 0.05, 0.10),
                                       chokepoint="CP_HORMUZ",
                                       duration_days=100,
                                       scenario_class="full_closure",
                                       target_count=15):
    """Re-solve the Hormuz closure with the disruption transition matrix made more or less adverse, testing the least data-informed block.

    Per delta, every row of the unresponsive matrix moves a fraction delta of its recovery mass into persistence and escalation (delta > 0 more adverse), reallocated by the responsive-matrix rule so rows stay distributions. The responsive matrix is rederived from the perturbed baseline, so exogenous and decision-dependent probabilities move together. Reports closed-to-closed persistence, no-intervention and optimized loss, saving, and active set. Baseline restored on exit.
    """
    regime = class_regime(scenario_class)[1]
    base = deepcopy(model._TRANSITIONS)
    network, scenarios, disrupted, unresp, solver = _hormuz_setup(
        scenario_class, chokepoint, target_count)

    def _loss(policy_subset):
        m = build_model(network, get_scenarios(scenario_class, target_count),
                        disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(policy_subset))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError("transition-perturbation solve infeasible")
        return m, float(info["objective"]) / 1000.0

    rows = []
    try:
        for d in deltas:
            # rho = -d: d > 0 scales adverse mass up by (1 + d) and draws the difference from recovery.
            model._TRANSITIONS[regime] = responsive_matrix(base[regime],
                                                                rho=-d)
            # Stage transitions come from the cached continuous-time generator (C8), so rebuild this regime's generator and drop the memoized stage matrices and scenario tree.
            model._GENERATORS[regime] = model._generator_from_matrix(
                model._TRANSITIONS[regime])
            model._stage_matrix.cache_clear()
            model._scenario_cache.clear()
            cc = model._TRANSITIONS[regime]["closed"]["closed"]
            _, no_int = _loss([])
            m_opt, opt = _loss(ALL_POLICIES)
            active = _active_set(m_opt)
            rows.append({
                "adverse shift": d,
                "closed->closed": round(cc, 3),
                "No intervention ($B)": round(no_int, 1),
                "Optimized ($B)": round(opt, 1),
                "Saving (%)": round(100.0 * (1.0 - opt / no_int), 1),
                "Active policies": "+".join(active),
            })
            print(f"  delta={d:+.2f} (cc={cc:.3f}): no-int={no_int:6.1f}B  "
                  f"opt={opt:6.1f}B  saving={100.0*(1.0-opt/no_int):4.1f}%  "
                  f"active={'+'.join(active)}")
            del m_opt
            gc.collect()
    finally:
        model._TRANSITIONS = base
        model._GENERATORS = {
            name: model._generator_from_matrix(rows)
            for name, rows in base.items()}
        model._stage_matrix.cache_clear()
        model._scenario_cache.clear()
    return pd.DataFrame(rows)


# 27-path loss distribution and CVaR
def _discrete_cvar(costs, probs, alpha):
    """CVaR at level alpha over a discrete loss distribution.

    Averages the worst (1 - alpha) probability mass, splitting the straddling scenario. Returns (VaR_alpha, CVaR_alpha).
    """
    tail = 1.0 - alpha
    order = sorted(costs, key=lambda w: costs[w], reverse=True)
    acc, cvar, var = 0.0, 0.0, costs[order[-1]]
    for w in order:
        take = min(probs[w], tail - acc)
        if take <= 0:
            break
        cvar += take * costs[w]
        var = costs[w]
        acc += take
        if acc >= tail - 1e-12:
            break
    return var, (cvar / tail if tail > 0 else costs[order[0]])


def attrition_ranking_stability(classes=("attrition_mild", "attrition_moderate",
                                         "attrition_severe"),
                                chokepoint="CP_HORMUZ",
                                duration_days=100,
                                target_count=15):
    """Solve the optimized response under each war-of-attrition regime, testing whether the instrument ranking is stable.

    One row per class: no-intervention and optimized loss, saving, and active bundle.
    """
    rows = []
    for cls in classes:
        network, scenarios, disrupted, unresp, solver = _hormuz_setup(
            cls, chokepoint, target_count)

        def _loss(policy_subset):
            m = build_model(network, scenarios, disrupted_chokepoint=disrupted,
                            duration_days=duration_days, enable_cascade=True,
                            endogenous_uncertainty=True,
                            unresponsive_regime=unresp,
                            policy_subset=list(policy_subset))
            info = solve_model(m, solver_name=solver)
            if info["objective"] is None:
                raise RuntimeError(f"attrition solve infeasible ({cls})")
            return m, float(info["objective"]) / 1000.0

        _, no_int = _loss([])
        m_opt, opt = _loss(ALL_POLICIES)
        active = _active_set(m_opt)
        rows.append({
            "Scenario": cls.replace("attrition_", "attrition ("),
            "No intervention ($B)": round(no_int, 1),
            "Optimized ($B)": round(opt, 1),
            "Saving (%)": round(100.0 * (1.0 - opt / no_int), 1),
            "Active policies": "+".join(active),
        })
        print(f"  {cls:20s}: no-int={no_int:6.1f}B  opt={opt:6.1f}B  "
              f"saving={100.0*(1.0-opt/no_int):4.1f}%  active={'+'.join(active)}")
        del m_opt
        gc.collect()
    return pd.DataFrame(rows)


def joint_uncertainty_analysis(n_draws=200,
                               chokepoint="CP_HORMUZ",
                               duration_days=100,
                               scenario_class="full_closure",
                               target_count=9,
                               seed=42):
    """Monte-Carlo JOINT sensitivity of the Hormuz closure.

    Samples EIGHT inputs jointly per draw and re-solves the no-intervention and optimized 100-day full-closure case, reporting the loss/saving distribution with CIs and per-policy selection frequency. Every perturbed calibration is restored on exit.

    Per draw, independent draws from a fixed-seed generator (default 42) for reproducibility:

      1. Cascade intensity: every beta entry scaled by Uniform(0.8, 1.2).
      2. Policy activation costs: every p.cost scaled by Uniform(0.5, 2.0).
      3. Import dependence: IMPORT_SHARE scaled by Uniform(0.85, 1.15), clamped to [0.05, 1.0].
      4. Emergency stocks: spr_days scaled by Uniform(0.8, 1.2).
      5. Global spare capacity: GLOBAL_SPARE_CAPACITY_MBD scaled by Uniform(0.5, 1.5).
      6. Oil demand elasticity: OIL_DEMAND_ELASTICITY drawn from Uniform(-0.30, -0.12).
      7. Transition rates: same reallocation as transition_perturbation_robustness, adverse shift Uniform(-0.10, 0.10).
      8. Emergency substitute supply: SUBSTITUTE_SUPPLY_FRAC scaled by Uniform(0.6, 1.6) (low/base/high ~3%/5%/8% of demand) and the three per-barrel channel costs scaled together by Uniform(0.8, 1.25), so P6 shares and costs are sampled directly, not only via activation cost.

    Knobs are read at build time from module globals or the live network, so each override reaches the model. Network and scenarios are built once; spr_days is perturbed in place on the cached country objects.

    Returns one row per completed draw. Summary stats (mean, median, std, p5, p95) for the two losses and saving are in df.attrs["summary"]; per-policy selection frequency in df.attrs["policy_frequency"].
    """
    rng = np.random.default_rng(seed)
    regime = class_regime(scenario_class)[1]

    # Baselines to restore on exit.
    base_beta = {s: dict(d) for s, d in model.BETA.items()}
    base_costs = {p.pid: p.cost for p in POLICIES}
    base_import = dict(model.IMPORT_SHARE)
    base_spare = model.GLOBAL_SPARE_CAPACITY_MBD
    base_elast = model.OIL_DEMAND_ELASTICITY
    base_transitions = deepcopy(model._TRANSITIONS)
    base_ss_frac = model.SUBSTITUTE_SUPPLY_FRAC
    base_ss_costs = (model.SUBSTITUTE_COST_PS, model.SUBSTITUTE_COST_RF,
                     model.SUBSTITUTE_COST_FS)

    network, scenarios, disrupted, unresp, solver = _hormuz_setup(
        scenario_class, chokepoint, target_count)
    # spr_days lives on the cached country objects, so save the baseline and perturb in place, restoring on exit.
    base_spr = {n: c.spr_days for n, c in network.countries.items()}

    def _loss(policy_subset):
        # get_scenarios rebuilt per draw because the transition perturbation clears the scenario cache.
        m = build_model(network, get_scenarios(scenario_class, target_count),
                        disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(policy_subset))
        info = solve_model(m, solver_name=solver)
        obj = info["objective"]
        if obj is None:
            return None, None
        return m, float(obj) / 1000.0

    rows, skipped = [], 0
    try:
        for i in range(n_draws):
            # 1. Cascade intensity.
            sc_beta = float(rng.uniform(0.8, 1.2))
            model.BETA = {s: {k: round(v * sc_beta, 4) for k, v in d.items()}
                               for s, d in base_beta.items()}
            # 2. Policy activation costs.
            for p in POLICIES:
                p.cost = base_costs[p.pid] * float(rng.uniform(0.5, 2.0))
            # 3. Import dependence.
            model.IMPORT_SHARE = {
                k: min(1.0, max(0.05, v * float(rng.uniform(0.85, 1.15))))
                for k, v in base_import.items()}
            # 4. Emergency stocks (perturbed in place on the network countries).
            for n, c in network.countries.items():
                c.spr_days = base_spr[n] * float(rng.uniform(0.8, 1.2))
            # 5. Global spare capacity.
            model.GLOBAL_SPARE_CAPACITY_MBD = base_spare * float(
                rng.uniform(0.5, 1.5))
            # 6. Oil demand elasticity.
            model.OIL_DEMAND_ELASTICITY = float(rng.uniform(-0.30, -0.12))
            # 7. Transition rates: same reallocation as the one-at-a-time check.
            d_shift = float(rng.uniform(-0.10, 0.10))
            model._TRANSITIONS[regime] = responsive_matrix(
                base_transitions[regime], rho=-d_shift)
            model._GENERATORS[regime] = model._generator_from_matrix(
                model._TRANSITIONS[regime])
            model._stage_matrix.cache_clear()
            model._scenario_cache.clear()
            # 8. Emergency substitute supply: SUBSTITUTE_SUPPLY_FRAC scaled by Uniform(0.6, 1.6) (~3%/5%/8% of demand), the three channel costs scaled together by Uniform(0.8, 1.25).
            sc_ss = float(rng.uniform(0.6, 1.6))
            model.SUBSTITUTE_SUPPLY_FRAC = base_ss_frac * sc_ss
            sc_sscost = float(rng.uniform(0.8, 1.25))
            model.SUBSTITUTE_COST_PS = base_ss_costs[0] * sc_sscost
            model.SUBSTITUTE_COST_RF = base_ss_costs[1] * sc_sscost
            model.SUBSTITUTE_COST_FS = base_ss_costs[2] * sc_sscost

            _, no_int = _loss([])
            m_opt, opt = _loss(ALL_POLICIES)
            if no_int is None or opt is None or no_int <= 0.0:
                skipped += 1
                if m_opt is not None:
                    del m_opt
                gc.collect()
                continue
            active = _active_set(m_opt)
            sav = 100.0 * (1.0 - opt / no_int)
            rows.append({
                "draw": i + 1,
                "beta scale": round(sc_beta, 3),
                "spare (Mbd)": round(model.GLOBAL_SPARE_CAPACITY_MBD, 3),
                "elasticity": round(model.OIL_DEMAND_ELASTICITY, 3),
                "adverse shift": round(d_shift, 3),
                "subst frac": round(model.SUBSTITUTE_SUPPLY_FRAC, 4),
                "subst cost scale": round(sc_sscost, 3),
                "No intervention ($B)": round(no_int, 1),
                "Optimized ($B)": round(opt, 1),
                "Saving (%)": round(sav, 1),
                "Active policies": "+".join(active),
                "_active": set(active),
            })
            if (i + 1) % 20 == 0 or (i + 1) == n_draws:
                print(f"  draw {i+1}/{n_draws}: no-int={no_int:.0f} "
                      f"opt={opt:.0f} save={sav:.1f}%")
            del m_opt
            gc.collect()
    finally:
        model.BETA = base_beta
        for p in POLICIES:
            p.cost = base_costs[p.pid]
        model.IMPORT_SHARE = base_import
        for n, c in network.countries.items():
            c.spr_days = base_spr[n]
        model.GLOBAL_SPARE_CAPACITY_MBD = base_spare
        model.OIL_DEMAND_ELASTICITY = base_elast
        model.SUBSTITUTE_SUPPLY_FRAC = base_ss_frac
        (model.SUBSTITUTE_COST_PS, model.SUBSTITUTE_COST_RF,
         model.SUBSTITUTE_COST_FS) = base_ss_costs
        model._TRANSITIONS = base_transitions
        model._GENERATORS = {
            name: model._generator_from_matrix(rws)
            for name, rws in base_transitions.items()}
        model._stage_matrix.cache_clear()
        model._scenario_cache.clear()

    active_sets = [r.pop("_active") for r in rows]
    df = pd.DataFrame(rows)

    def _stats(vals):
        a = np.asarray(vals, dtype=float)
        return {"mean": round(float(a.mean()), 2),
                "median": round(float(np.median(a)), 2),
                "std": round(float(a.std(ddof=1)) if a.size > 1 else 0.0, 2),
                "p5": round(float(np.percentile(a, 5)), 2),
                "p95": round(float(np.percentile(a, 95)), 2)}

    summary = {}
    if rows:
        summary = {
            "No intervention ($B)": _stats(df["No intervention ($B)"]),
            "Optimized ($B)": _stats(df["Optimized ($B)"]),
            "Saving (%)": _stats(df["Saving (%)"]),
        }
    n_done = len(rows)
    policy_freq = {pid: (round(sum(1 for a in active_sets if pid in a) / n_done, 3)
                         if n_done else 0.0)
                   for pid in ALL_POLICIES}
    df.attrs["summary"] = summary
    df.attrs["policy_frequency"] = policy_freq
    df.attrs["n_draws"] = n_done
    df.attrs["n_skipped"] = skipped

    if summary:
        s = summary["Saving (%)"]
        print(f"  saving: mean={s['mean']:.1f}%  "
              f"[p5={s['p5']:.1f}, p95={s['p95']:.1f}]")
        ranked = sorted(policy_freq.items(), key=lambda kv: kv[1], reverse=True)
        freq_txt = "  ".join(f"{pid}={fr*100:.0f}%" for pid, fr in ranked)
        print(f"  policy-selection frequency: {freq_txt}")
    print(f"  draws completed={n_done}  skipped={skipped}")
    return df


def loss_distribution_cvar(chokepoint="CP_HORMUZ",
                           duration_days=100,
                           scenario_class="full_closure",
                           target_count=15,
                           alphas=(0.90, 0.95)):
    """Solve the Hormuz closure under no intervention and the optimized portfolio, reporting the total-cost distribution across enumerated paths rather than the expectation alone.

    Per regime, from the per-scenario cost and realized path probabilities: prob-weighted mean, std, min, max, and VaR/CVaR at each alpha, exposing the tail the expected-cost objective averages over. No calibration changed.
    """
    network, scenarios, disrupted, unresp, solver = _hormuz_setup(
        scenario_class, chokepoint, target_count)

    def _solve(policy_subset):
        m = build_model(network, get_scenarios(scenario_class, target_count),
                        disrupted_chokepoint=disrupted,
                        duration_days=duration_days, enable_cascade=True,
                        endogenous_uncertainty=True, unresponsive_regime=unresp,
                        policy_subset=list(policy_subset))
        info = solve_model(m, solver_name=solver)
        if info["objective"] is None:
            raise RuntimeError("loss-distribution solve infeasible")
        probs = realized_scenario_probs(m)
        costs = {w: float(pyo.value(m.scenario_cost[w])) / 1000.0 for w in m.OMEGA}
        return m, probs, costs

    rows = []
    for label, subset in (("No intervention", []),
                          ("Optimized", ALL_POLICIES)):
        m, probs, costs = _solve(subset)
        mean = sum(probs[w] * costs[w] for w in costs)
        var_ = sum(probs[w] * (costs[w] - mean) ** 2 for w in costs)
        std = var_ ** 0.5
        lo = min(costs.values())
        hi = max(costs.values())
        row = {"Regime": label,
               "Paths": len(costs),
               "Expected ($B)": round(mean, 1),
               "Std ($B)": round(std, 1),
               "Min ($B)": round(lo, 1),
               "Max ($B)": round(hi, 1)}
        for a in alphas:
            v, cv = _discrete_cvar(costs, probs, a)
            row[f"VaR{int(a*100)} ($B)"] = round(v, 1)
            row[f"CVaR{int(a*100)} ($B)"] = round(cv, 1)
        rows.append(row)
        tail_txt = "  ".join(f"CVaR{int(a*100)}={row[f'CVaR{int(a*100)} ($B)']}"
                             for a in alphas)
        print(f"  {label:16s} E={mean:6.1f}B  sd={std:5.1f}  "
              f"[{lo:.1f}, {hi:.1f}]  {tail_txt}")
        del m
        gc.collect()
    return pd.DataFrame(rows)
