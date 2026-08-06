"""Consolidated experiment driver.

Single argparse CLI over the paper's re-run and robustness experiments. Each
subcommand reproduces one experiment and writes its outputs under outputs/.
Configuration is read from environment variables so the HPC job scripts keep working.

Run from the repository root, e.g.:
  COALITION_TARGET_COUNT=30 python code/experiments.py reconcile
  SHARD_INDEX=1 SHARD_COUNT=8 python code/experiments.py core-shard
  python code/experiments.py --help

Subcommands:
  reconcile       three-regime reconciliation + computational disclosure
  coalition       coordination re-solve, physical clearing and normalized game
  ampspec         amplification-specification benchmark
  displacement    reserve/substitute market-displacement sensitivity
  isoelastic      constant-elasticity inverse-demand benchmark
  isogame         constant-elasticity cooperative game and core
  grid-refine     deadweight-grid refinement for the isoelastic benchmark
  stagelen        stage-length (decision-frequency) sensitivity
  core-shard      one shard of the certified core-stability re-run (HPC fan-out)
  core-combine    combine the core-certification shards into the final game
  core-certify    certified core-stability of the cooperative game (single node)
  robustness      full robustness and sensitivity sweep
  build-cascade   rebuild the six-sector cascade matrix beta from WIOD tables
"""
import argparse
import os
import sys
import json
import time
import glob
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("GEOENERGY_SOLVER", "appsi_gurobi")

import pyomo.environ as pyo  # noqa: E402
import model as mc  # noqa: E402
import analysis as an  # noqa: E402


# Shared helpers
def _active_bundle(m):
    """Sorted "+"-joined list of policies switched on in any stage/scenario."""
    on = [p for p in m.P
          if any(round(pyo.value(m.z[p, t, w])) >= 1 for t in m.T for w in m.OMEGA)]
    return "+".join(sorted(on)) if on else "(none)"


def _obj_billion(m):
    """Objective (in $M) as $B."""
    return pyo.value(list(m.component_data_objects(pyo.Objective))[0]) / 1e3


# reconcile: three-regime reconciliation + computational disclosure
def cmd_reconcile(args):
    """Solve the 100-day full-closure Hormuz benchmark in three regimes, reporting per
    regime the active bundle and welfare decomposition: real resource loss net of price
    deadweight (W'), deadweight, terms-of-trade transfer, importer expenditure, plus
    problem dimensions, incumbent, bound, gap, and solve time. Separates the resource
    saving (no intervention -> planner) from the coordination gain (atomistic -> planner).
    Writes outputs/table_reconcile.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    TL = int(os.environ.get("RECON_TIME_LIMIT", "1200"))
    _SOLVER = mc._detect_solver(None)
    print(f"[config] target_count={TC} time_limit={TL} solver={_SOLVER}", flush=True)

    def _build(policy_subset, coord):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        return mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                              unresponsive_regime=mc.class_regime("full_closure")[1],
                              policy_subset=list(policy_subset), coordinated_price=coord)

    def _dims(m):
        nv = sum(1 for _ in m.component_data_objects(pyo.Var))
        nb = sum(1 for v in m.component_data_objects(pyo.Var) if v.is_binary())
        nc = sum(1 for _ in m.component_data_objects(pyo.Constraint, active=True))
        return nv, nb, nc

    REGIMES = [("No intervention", (), True),
               ("Atomistic response", mc.ALL_POLICIES, False),
               ("Coordinated planner", mc.ALL_POLICIES, True)]

    rows = []
    for name, pol, coord in REGIMES:
        m = _build(pol, coord)
        t0 = time.time()
        res = mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=1e-3)
        dt = time.time() - t0
        b = an._coordination_bundle(m)
        wprime = b["efficiency"] - b["deadweight"]
        obj = pyo.value(list(m.component_data_objects(pyo.Objective))[0]) / 1e3
        try:
            lb = float(res["result"].problem.lower_bound) / 1e3
            ub = float(res["result"].problem.upper_bound) / 1e3
            gap = 100.0 * abs(ub - lb) / max(abs(ub), 1e-9)
        except Exception:
            lb, gap = None, None
        row = dict(regime=name, bundle=_active_bundle(m), wprime=round(wprime, 1),
                   dwl=round(b["deadweight"], 1), transfer=round(b["transfer"], 1),
                   importer=round(b["importer"], 1), resource=round(b["efficiency"], 1),
                   obj=round(obj, 1), bound=(round(lb, 1) if lb is not None else None),
                   gap_pct=(round(gap, 3) if gap is not None else None), time_s=round(dt, 1))
        if name == "Coordinated planner":
            nv, nb, nc = _dims(m)
            row["nvars"], row["nbin"], row["ncons"] = nv, nb, nc
        rows.append(row)
        print(f"  {name}: bundle={row['bundle']} W'={row['wprime']} DWL={row['dwl']} "
              f"transfer={row['transfer']} importer={row['importer']} resource={row['resource']} "
              f"obj={row['obj']} bound={row['bound']} gap={row['gap_pct']}% time={row['time_s']}s",
              flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table_reconcile.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== reconciliation rows: regime & bundle & W' & deadweight & transfer & importer exp ===", flush=True)
    for r in rows:
        print(f"  {r['regime']:22s} & {r['bundle']} & {r['wprime']} & {r['dwl']} "
              f"& {r['transfer']} & {r['importer']} \\\\", flush=True)
    pl = [r for r in rows if r["regime"] == "Coordinated planner"][0]
    print(f"\nDIMENSIONS (planner): vars={pl.get('nvars')} binaries={pl.get('nbin')} "
          f"constraints={pl.get('ncons')}", flush=True)
    print("SOLVE (obj / bound / gap% / time s): "
          + " | ".join(f"{r['regime']}={r['obj']}/{r['bound']}/{r['gap_pct']}/{r['time_s']}"
                       for r in rows), flush=True)
    print("Wrote outputs/table_reconcile.json", flush=True)


# coalition: coordination re-solve (physical clearing + normalized game)
def cmd_coalition(args):
    """Regenerate the decentralization/coordination results. Deadweight and transfer use
    the physical price residual netting out every realized demand reduction; the game is
    normalized with v_+(S)=max{0,v(S)}. Writes outputs/table_coordination_analysis.csv,
    table_coalition_cost_sharing.csv, coalition_game_summary.json, and
    table_coalition_transfers.csv."""
    import pandas as pd  # noqa: F401  (kept for parity with the original script)

    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    print(f"[config] target_count={TC}", flush=True)
    os.makedirs("outputs", exist_ok=True)

    # Coordination analysis: planner vs atomistic, physical deadweight/transfer.
    print("\n########## coordination analysis (physical residual) ##########", flush=True)
    df_coord = an.coordination_analysis(target_count=TC)
    df_coord.to_csv("outputs/table_coordination_analysis.csv", index=False)
    print(df_coord.to_string(index=False), flush=True)

    # Normalized cooperative game: physical DT with v_+ = max{0,v}.
    print("\n########## normalized cooperative game ##########", flush=True)
    cs = an.coalition_cost_sharing(target_count=TC)
    cs.to_csv("outputs/table_coalition_cost_sharing.csv", index=False)
    with open("outputs/coalition_game_summary.json", "w") as fh:
        json.dump({k: cs.attrs[k] for k in ("grand_coalition_value", "shapley_sum",
                                            "core_stable", "min_proper_core_excess",
                                            "closest_blocking_coalition") if k in cs.attrs},
                  fh, indent=2)
    print(cs.to_string(index=False), flush=True)
    print(f"v(N)={cs.attrs.get('grand_coalition_value')}  "
          f"min_proper_core_excess={cs.attrs.get('min_proper_core_excess')} "
          f"at {cs.attrs.get('closest_blocking_coalition')}  "
          f"core_stable={cs.attrs.get('core_stable')}", flush=True)

    # Budget-balanced transfer table.
    print("\n########## cost-sharing transfer table ##########", flush=True)
    shapley = dict(zip(cs["Bloc"], cs["Shapley allocation ($B)"]))
    df_tr = an.coalition_transfer_table(shapley, target_count=TC)
    df_tr.to_csv("outputs/table_coalition_transfers.csv", index=False)
    print(df_tr.to_string(index=False), flush=True)
    print(f"dt_saved={df_tr.attrs.get('dt_saved')}  "
          f"transfer_sum={df_tr.attrs.get('transfer_sum')}  "
          f"shapley_sum={df_tr.attrs.get('shapley_sum')}", flush=True)
    print("\nWrote coordination, game, and transfer outputs.", flush=True)


# ampspec: amplification-specification benchmark
def cmd_ampspec(args):
    """Re-solve the 100-day full-closure Hormuz benchmark under three amplification
    specifications, reporting per spec the active bundle, welfare saving, amplification
    share of the no-intervention loss, the three most-affected importers, and the value
    of acting early rather than after a one-stage delay. Writes outputs/table_ampspec.json.

      headline : normalized first-pass, critical-service multiplier eta_s = g_s(1)
      capped   : eta_s = min(1, g_s(1)), loss bounded by own value added
      leontief : full total-requirements cascade (I - beta)^{-1}"""
    TC = int(os.environ.get("AMPSPEC_TC", os.environ.get("COALITION_TARGET_COUNT", "30")))
    TL = int(os.environ.get("AMPSPEC_TIME_LIMIT", "900"))
    print(f"[config] target_count={TC} time_limit={TL}", flush=True)

    SPECS = [("headline", "Normalized first-pass (headline)"),
             ("capped",   "Capped intensity eta_s<=1"),
             ("leontief", "Full Leontief inverse")]

    _SOLVER = mc._detect_solver(None)

    def _solve(policy_subset, amp_spec, delay=0):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        m = mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                           unresponsive_regime=mc.class_regime("full_closure")[1],
                           policy_subset=list(policy_subset), amp_spec=amp_spec)
        if delay > 0:
            for p in policy_subset:
                for t in range(delay):
                    for w in m.OMEGA:
                        m.z[p, t, w].fix(0)
        mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=1e-3)
        return m

    rows = []
    for spec_key, spec_name in SPECS:
        print(f"\n=== {spec_name} ===", flush=True)
        m0 = _solve((), spec_key)
        mO = _solve(mc.ALL_POLICIES, spec_key)
        mD = _solve(mc.ALL_POLICIES, spec_key, delay=1)  # one-stage delay
        o0, oO, oD = _obj_billion(m0), _obj_billion(mO), _obj_billion(mD)
        saving = 100.0 * (o0 - oO) / o0 if o0 else float("nan")
        comps = an.cost_components(m0)
        amp_share = 100.0 * comps.get("Cascade", 0.0) / comps["Total"] if comps.get("Total") else float("nan")
        losses = {code: sum(v["total"] for v in an.sector_damage(m0, code).values()) for code in m0.C}
        top = sorted(losses, key=losses.get, reverse=True)[:3]
        bundle = _active_bundle(mO)
        early = oD - oO  # extra loss from delaying the optimized bundle one stage
        row = dict(spec=spec_name, bundle=bundle, saving=round(saving, 1),
                   amp_share=round(amp_share), top=top, early=round(early, 1),
                   noint=round(o0, 1), opt=round(oO, 1))
        rows.append(row)
        print(f"  no-int=${o0:.1f}B  opt=${oO:.1f}B  saving={saving:.1f}%  "
              f"amp_share={amp_share:.0f}%  bundle={bundle}  top={top}  early=${early:.1f}B",
              flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table_ampspec.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== tab_ampspec rows (paste into the paper) ===", flush=True)
    for r in rows:
        print(f"  {r['spec']:34s} & {r['bundle']} & {r['saving']} & {r['amp_share']} "
              f"& {r['early']} \\\\   [top: {', '.join(r['top'])}]", flush=True)
    print("Wrote outputs/table_ampspec.json", flush=True)


# displacement: reserve/substitute market-displacement sensitivity
def cmd_displacement(args):
    """Re-solve the 100-day full-closure Hormuz headline (no intervention vs optimal
    portfolio) under three displacement coefficients phi applied uniformly to reserve
    release and the three substitute channels (product stock, refinery yield, fuel switch).
    A positive phi lets that share moderate the world price like conservation, lowering the
    optimized deadweight and raising the saving; the no-intervention case is invariant to
    phi. Writes outputs/table_displacement.json.

      zero    phi = 0.0   domestic coping displaces no world-market purchase (the headline)
      partial phi = 0.5   half of each channel's volume also reduces world demand
      full    phi = 1.0   one-for-one displacement of world-market purchases"""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    TL = int(os.environ.get("DISP_TIME_LIMIT", "900"))
    print(f"[config] target_count={TC} time_limit={TL}", flush=True)

    CASES = [("zero", 0.0), ("partial", 0.5), ("full", 1.0)]
    _SOLVER = mc._detect_solver(None)

    def _disp(phi):
        return {"reserve": phi, "product_stock": phi, "refinery": phi, "fuel_switch": phi}

    def _solve(policy_subset, phi):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        m = mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                           unresponsive_regime=mc.class_regime("full_closure")[1],
                           policy_subset=list(policy_subset), displacement=_disp(phi))
        mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=1e-3)
        return m

    rows = []
    for name, phi in CASES:
        print(f"\n=== displacement {name} (phi={phi}) ===", flush=True)
        m0 = _solve((), phi)
        mO = _solve(mc.ALL_POLICIES, phi)
        o0, oO = _obj_billion(m0), _obj_billion(mO)
        saving = 100.0 * (o0 - oO) / o0 if o0 else float("nan")
        bundle = _active_bundle(mO)
        row = dict(case=name, phi=phi, noint=round(o0, 1), opt=round(oO, 1),
                   saving=round(saving, 1), bundle=bundle)
        rows.append(row)
        print(f"  no-int=${o0:.1f}B  opt=${oO:.1f}B  saving={saving:.1f}%  bundle={bundle}",
              flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table_displacement.json", "w") as fh:
        json.dump(rows, fh, indent=2)

    print("\n=== displacement rows (paste into the paper) ===", flush=True)
    for r in rows:
        print(f"  {r['case']:8s} phi={r['phi']} & {r['noint']} & {r['opt']} "
              f"& {r['saving']}\\% & {r['bundle']} \\\\", flush=True)
    print("Wrote outputs/table_displacement.json", flush=True)


# isoelastic: constant-elasticity inverse-demand benchmark
def cmd_isoelastic(args):
    """Re-solve the 100-day full-closure Hormuz benchmark in three regimes under a
    constant-elasticity inverse demand P(q)=P0 (q/Qbar)^{-1/|eta|} instead of the linear
    approximation. The market clears at the same reduced quantity either way, so only
    price, deadweight, and terms-of-trade transfer change. Reports per regime the optimized
    welfare loss, active bundle, physical deadweight, and transfer. Writes
    outputs/table_isoelastic.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    TL = int(os.environ.get("ISO_TIME_LIMIT", "1200"))
    _SOLVER = mc._detect_solver(None)
    EPS = 1.0 / abs(mc.OIL_DEMAND_ELASTICITY)
    P0 = mc.BASELINE_OIL_PRICE
    QBAR = float(mc.WORLD_OIL_SUPPLY_MBD * mc.STAGE_DAYS)
    print(f"[config] TC={TC} TL={TL} solver={_SOLVER} eps={EPS} (|eta|={abs(mc.OIL_DEMAND_ELASTICITY)})", flush=True)

    def _build(policy_subset, coord):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        return mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                              unresponsive_regime=mc.class_regime("full_closure")[1],
                              policy_subset=list(policy_subset), coordinated_price=coord,
                              price_curve="isoelastic")

    def _iso_price_terms(m):
        """Physical isoelastic deadweight and transfer (USD billion) from price_loss_phys."""
        pw = an.realized_scenario_probs(m)
        Q0 = {t: sum(pyo.value(m.demand[n, s, t]) for n in m.C for s in m.S) for t in m.T}
        dwl = transfer = 0.0
        for t in m.T:
            for w in m.OMEGA:
                pl = pyo.value(m.price_loss_phys[t, w])
                f = min(pl / QBAR, 0.95)
                dwl_ce = P0 * Q0[t] * ((1.0 - (1.0 - f) ** (1.0 - EPS)) / (1.0 - EPS) - f)
                dP = P0 * ((1.0 - f) ** (-EPS) - 1.0)
                dwl += pw[w] * dwl_ce / 1e3
                transfer += pw[w] * dP * Q0[t] * (1.0 - f) / 1e3
        return dwl, transfer

    REGIMES = [("No intervention", (), True),
               ("Atomistic response", mc.ALL_POLICIES, False),
               ("Coordinated planner", mc.ALL_POLICIES, True)]

    rows = []
    for name, pol, coord in REGIMES:
        m = _build(pol, coord)
        mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=1e-3)
        obj = pyo.value(list(m.component_data_objects(pyo.Objective))[0]) / 1e3
        dwl, transfer = _iso_price_terms(m)
        importer = obj + transfer
        row = dict(regime=name, bundle=_active_bundle(m), obj=round(obj, 1),
                   dwl=round(dwl, 1), transfer=round(transfer, 1), importer=round(importer, 1))
        rows.append(row)
        print(f"  {name}: bundle={row['bundle']} obj={row['obj']} deadweight={row['dwl']} "
              f"transfer={row['transfer']} importer={row['importer']}", flush=True)

    noint = rows[0]["obj"]
    plan = rows[2]["obj"]
    saving = 100.0 * (noint - plan) / noint if noint else float("nan")
    coord_gain = rows[1]["importer"] - rows[2]["importer"]
    print(f"\nISOELASTIC saving (no-int -> planner) = {saving:.1f}%  "
          f"coordination gain (atomistic -> planner importer welfare) = {coord_gain:.1f}", flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/table_isoelastic.json", "w") as fh:
        json.dump({"rows": rows, "saving_pct": round(saving, 1),
                   "coord_gain": round(coord_gain, 1)}, fh, indent=2)
    print("Wrote outputs/table_isoelastic.json", flush=True)


# isogame: constant-elasticity cooperative game and core
def cmd_isogame(args):
    """Recompute the six-bloc cooperative game under constant-elasticity inverse demand to
    compare the Shapley allocation, minimum proper-coalition core excess, and stand-alone
    values against the linear benchmark. The larger isoelastic price channel shifts the
    grand-coalition value and allocation while the policy bundles stay stable. Writes
    outputs/table_coalition_isoelastic.csv and coalition_game_isoelastic.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    print(f"[config] target_count={TC} price_curve=isoelastic", flush=True)
    os.makedirs("outputs", exist_ok=True)

    cs = an.coalition_cost_sharing(target_count=TC, price_curve="isoelastic")
    cs.to_csv("outputs/table_coalition_isoelastic.csv", index=False)
    with open("outputs/coalition_game_isoelastic.json", "w") as fh:
        json.dump({k: cs.attrs[k] for k in ("grand_coalition_value", "shapley_sum",
                                            "core_stable", "min_proper_core_excess",
                                            "closest_blocking_coalition") if k in cs.attrs},
                  fh, indent=2)
    print(cs.to_string(index=False), flush=True)
    print(f"v(N)={cs.attrs.get('grand_coalition_value')}  "
          f"min_proper_core_excess={cs.attrs.get('min_proper_core_excess')} "
          f"at {cs.attrs.get('closest_blocking_coalition')}  "
          f"core_stable={cs.attrs.get('core_stable')}", flush=True)
    print("Wrote outputs/table_coalition_isoelastic.csv", flush=True)


# grid-refine: deadweight-grid refinement for the isoelastic benchmark
def cmd_grid_refine(args):
    """Re-solve the isoelastic three-regime benchmark at the default 12-breakpoint and a
    doubled 24-breakpoint deadweight tangent grid, reporting the headline saving under each
    and their difference to certify that the piecewise-linear selection error is negligible.
    The quantity clears identically either way, so only the objective's tangent
    under-approximation changes. Writes outputs/grid_refine.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    TL = int(os.environ.get("ISO_TIME_LIMIT", "1200"))
    GAP = float(os.environ.get("GRID_MIP_GAP", "1e-4"))
    _SOLVER = mc._detect_solver(None)
    EPS = 1.0 / abs(mc.OIL_DEMAND_ELASTICITY)
    P0 = mc.BASELINE_OIL_PRICE
    QBAR = float(mc.WORLD_OIL_SUPPLY_MBD * mc.STAGE_DAYS)
    print(f"[config] TC={TC} TL={TL} gap={GAP} solver={_SOLVER} eps={EPS}", flush=True)

    REGIMES = [("No intervention", (), True),
               ("Atomistic response", mc.ALL_POLICIES, False),
               ("Coordinated planner", mc.ALL_POLICIES, True)]

    def _build(policy_subset, coord):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        return mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                              unresponsive_regime=mc.class_regime("full_closure")[1],
                              policy_subset=list(policy_subset), coordinated_price=coord,
                              price_curve="isoelastic")

    def _saving_for_grid(nbp):
        os.environ["GEOENERGY_DWL_NBP"] = str(nbp)
        objs = {}
        for name, pol, coord in REGIMES:
            m = _build(pol, coord)
            mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=GAP)
            objs[name] = pyo.value(list(m.component_data_objects(pyo.Objective))[0]) / 1e3
            del m
        noint, plan = objs["No intervention"], objs["Coordinated planner"]
        saving = 100.0 * (noint - plan) / noint if noint else float("nan")
        print(f"  [NBP={nbp}] no-int={noint:.2f}B planner={plan:.2f}B saving={saving:.2f}%", flush=True)
        return saving, objs

    res = {}
    for nbp in (12, 24):
        s, o = _saving_for_grid(nbp)
        res[nbp] = {"saving_pct": round(s, 3), "objs": {k: round(v, 2) for k, v in o.items()}}

    diff = abs(res[24]["saving_pct"] - res[12]["saving_pct"])
    print(f"\nGRID REFINEMENT: saving(12bp)={res[12]['saving_pct']}%  "
          f"saving(24bp)={res[24]['saving_pct']}%  |diff|={diff:.3f} point", flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/grid_refine.json", "w") as fh:
        json.dump({"by_nbp": res, "abs_saving_diff_points": round(diff, 3)}, fh, indent=2)
    print("Wrote outputs/grid_refine.json", flush=True)


# stagelen: stage-length (decision-frequency) sensitivity
def cmd_stagelen(args):
    """Re-solve the linear headline three-regime full-closure benchmark at stage lengths
    of 20, 25, and 30 days (100-day horizon fixed), reporting the saving and active bundle
    under each. Per-stage flow capacities scale with stage length while the barrel-fixed
    product-stock inventory stays fixed, so the instrument ranking and saving should be
    stable across decision frequency. Writes outputs/stage_length_sensitivity.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    TL = int(os.environ.get("STAGE_TIME_LIMIT", "1200"))
    GAP = float(os.environ.get("STAGE_MIP_GAP", "1e-4"))
    _SOLVER = mc._detect_solver(None)
    STAGE_LENGTHS = [int(x) for x in os.environ.get("STAGE_LENGTHS", "20,25,30").split(",")]
    print(f"[config] TC={TC} TL={TL} gap={GAP} solver={_SOLVER} stage_lengths={STAGE_LENGTHS}", flush=True)

    REGIMES = [("No intervention", (), True),
               ("Atomistic response", mc.ALL_POLICIES, False),
               ("Coordinated planner", mc.ALL_POLICIES, True)]

    def _build(policy_subset, coord, sd):
        net = mc.get_network()
        scen = mc.get_scenarios("full_closure", TC, stage_days=sd)
        disr = mc.disrupted_chokepoints_for_class("full_closure", "CP_HORMUZ")
        return mc.build_model(net, scen, disrupted_chokepoint=disr, duration_days=100,
                              unresponsive_regime=mc.class_regime("full_closure")[1],
                              policy_subset=list(policy_subset), coordinated_price=coord,
                              stage_days=sd)

    rows = []
    for sd in STAGE_LENGTHS:
        objs, bundles = {}, {}
        for name, pol, coord in REGIMES:
            m = _build(pol, coord, sd)
            mc.solve_model(m, solver_name=_SOLVER, time_limit=TL, mip_gap=GAP)
            objs[name] = pyo.value(list(m.component_data_objects(pyo.Objective))[0]) / 1e3
            bundles[name] = _active_bundle(m)
            del m
        noint, plan = objs["No intervention"], objs["Coordinated planner"]
        saving = 100.0 * (noint - plan) / noint if noint else float("nan")
        row = dict(stage_days=sd, no_int=round(noint, 1), planner=round(plan, 1),
                   saving_pct=round(saving, 2), planner_bundle=bundles["Coordinated planner"],
                   atomistic_bundle=bundles["Atomistic response"])
        rows.append(row)
        print(f"  [SD={sd}] no-int={noint:.1f}B planner={plan:.1f}B saving={saving:.2f}% "
              f"bundle={bundles['Coordinated planner']}", flush=True)

    print("\nSTAGE-LENGTH SENSITIVITY (saving % should be stable):", flush=True)
    for r in rows:
        print(f"  SD={r['stage_days']}d: saving={r['saving_pct']}%  bundle={r['planner_bundle']}", flush=True)

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/stage_length_sensitivity.json", "w") as fh:
        json.dump({"rows": rows}, fh, indent=2)
    print("Wrote outputs/stage_length_sensitivity.json", flush=True)


# core-shard: one shard of the certified core-stability re-run (HPC fan-out)
def cmd_core_shard(args):
    """Solve one shard of the certified core-stability re-run for HPC fan-out. The 63
    non-empty bloc coalitions are enumerated in fixed order and partitioned across
    SHARD_COUNT jobs by index. Every shard also re-solves the (deterministic, seed-fixed)
    no-intervention baseline for the realized shares and reference world-price cost, so
    each shard is self-contained and the combined result equals a single-node run. Writes
    outputs/shards/core_shard_{index:02d}.json.

    Environment:
      SHARD_INDEX (1-based, = LSB_JOBINDEX)   SHARD_COUNT (default 8)
      COALITION_TARGET_COUNT (30)   CORE_MIP_GAP (1e-4)   CORE_TIME_LIMIT (3600)
      PRICE_CURVE (linear | isoelastic)"""
    SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "1"))     # 1-based
    SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "8"))
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    GAP = float(os.environ.get("CORE_MIP_GAP", "1e-4"))
    TL = int(os.environ.get("CORE_TIME_LIMIT", "3600"))
    CURVE = os.environ.get("PRICE_CURVE", "linear")
    CHOKE, SCLASS, DUR = "CP_HORMUZ", "full_closure", 100

    solver = an._detect_solver(None)
    print(f"[shard {SHARD_INDEX}/{SHARD_COUNT}] TC={TC} gap={GAP} curve={CURVE} solver={solver}",
          flush=True)

    net = an.get_network()
    scen = an.get_scenarios(SCLASS, TC)
    disr = an.disrupted_chokepoints_for_class(SCLASS, CHOKE)
    unresp = an.class_regime(SCLASS)[1]

    blocs = list(an.REGIONAL_BLOCS.keys()) + ["Rest"]
    members = {b: [c for c in net.countries if an._bloc_of(c) == b] for b in blocs}
    demand = {c: net.countries[c].daily_oil_mbd for c in net.countries}
    tot_demand = sum(demand.values())
    base_share = {b: sum(demand[c] for c in members[b]) / tot_demand for b in blocs}

    def _solve(coord_codes, want_shares=False):
        m = an.build_model(net, scen, disrupted_chokepoint=disr, duration_days=DUR,
                           unresponsive_regime=unresp, policy_subset=list(an.ALL_POLICIES),
                           coordinated_price=bool(coord_codes) or coord_codes == "ALL",
                           coordinating_countries=(None if coord_codes == "ALL" else coord_codes),
                           price_curve=CURVE)
        res = an.solve_model(m, solver_name=solver, time_limit=TL, mip_gap=GAP)
        b = an._coordination_bundle(m)
        b["_gap"] = res.get("gap")
        sh = an._realized_bloc_shares(m) if want_shares else None
        del m
        an.gc.collect()
        return (b, sh) if want_shares else b

    # No-intervention baseline (empty coalition): realized shares and reference cost.
    nc, real_share = _solve([], want_shares=True)
    share = {b: real_share.get(b, base_share[b]) for b in blocs}
    dt_nc = nc["dt"]
    wprime_nc = nc["efficiency"] - nc["deadweight"]

    # Enumerate the 63 non-empty coalitions in fixed order, partition by shard index.
    all_S = [S for r in range(1, len(blocs) + 1) for S in combinations(blocs, r)]
    my_S = [S for i, S in enumerate(all_S) if i % SHARD_COUNT == (SHARD_INDEX - 1)]
    print(f"[shard {SHARD_INDEX}] solving {len(my_S)} of {len(all_S)} coalitions", flush=True)

    gaps = [nc["_gap"]] if nc.get("_gap") is not None else []
    v_raw = {}
    for S in my_S:
        codes = [c for b in S for c in members[b]]
        b = _solve(codes)
        if b.get("_gap") is not None:
            gaps.append(b["_gap"])
        share_S = sum(share[x] for x in S)
        wprime_S = b["efficiency"] - b["deadweight"]
        vr = share_S * (dt_nc - b["dt"]) - (wprime_S - wprime_nc)
        v_raw["+".join(sorted(S))] = round(float(vr), 6)
        print(f"  v({'+'.join(S)}) raw={vr:.3f}", flush=True)

    out = {
        "shard_index": SHARD_INDEX, "shard_count": SHARD_COUNT, "price_curve": CURVE,
        "mip_gap_target": GAP, "blocs": blocs,
        "share": {b: round(share[b], 6) for b in blocs},
        "dt_nc": round(float(dt_nc), 6), "wprime_nc": round(float(wprime_nc), 6),
        "v_raw": v_raw, "max_gap": (round(max(gaps), 8) if gaps else None),
    }
    os.makedirs("outputs/shards", exist_ok=True)
    path = f"outputs/shards/core_shard_{SHARD_INDEX:02d}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[shard {SHARD_INDEX}] wrote {path}  max_gap={out['max_gap']}", flush=True)


# core-combine: combine the core-certification shards into the final game
def cmd_core_combine(args):
    """Read every outputs/shards/core_shard_*.json from the LSF fan-out, assemble the full
    characteristic function, and compute the Shapley allocation, individual rationality,
    minimum proper-coalition core excess, and the certification (max achieved gap and the
    resulting numerical-uncertainty scale). Matches a single-node coalition_cost_sharing
    run. Writes outputs/table_coalition_certify_{tag}.csv and core_certify_{tag}.json.

    Environment: PRICE_CURVE (linear | isoelastic)"""
    import pandas as pd  # noqa: E402

    CURVE = os.environ.get("PRICE_CURVE", "linear")
    tag = "linear" if CURVE == "linear" else CURVE

    files = sorted(glob.glob("outputs/shards/core_shard_*.json"))
    if not files:
        sys.exit("No shard files found in outputs/shards/. Run the shard array first.")
    shards = [json.load(open(f)) for f in files]
    shards = [s for s in shards if s.get("price_curve", "linear") == CURVE]
    if not shards:
        sys.exit(f"No shards for price_curve={CURVE}.")

    base = shards[0]
    blocs = base["blocs"]
    share = base["share"]
    dt_nc = base["dt_nc"]

    # Shards re-solve the baseline independently; warn if the reference cost drifts more
    # than 0.1% across them, which would signal a nondeterministic baseline.
    _dt_vals = [s["dt_nc"] for s in shards]
    _spread = (max(_dt_vals) - min(_dt_vals)) / max(abs(dt_nc), 1e-9)
    if _spread > 1e-3:
        print(f"WARNING: baseline dt_nc varies {_spread:.2%} across shards "
              f"(min {min(_dt_vals):.3f}, max {max(_dt_vals):.3f}); "
              f"v(S) values carry this inconsistency.", flush=True)
    else:
        print(f"[check] baseline consistent across {len(shards)} shards "
              f"(dt_nc spread {_spread:.2e}).", flush=True)

    # Merge raw coalition values and achieved gaps.
    v_raw = {}
    gaps = []
    for s in shards:
        v_raw.update(s["v_raw"])
        if s.get("max_gap") is not None:
            gaps.append(s["max_gap"])

    # Characteristic function with zero outside option: v(S) = max(0, v_raw(S)).
    value_fn = {(): 0.0}
    for key, vr in v_raw.items():
        value_fn[tuple(sorted(key.split("+")))] = max(0.0, float(vr))

    all_S = [tuple(sorted(S)) for r in range(1, len(blocs) + 1) for S in combinations(blocs, r)]
    missing = [S for S in all_S if S not in value_fn]
    if missing:
        sys.exit(f"Incomplete shards, missing {len(missing)} coalitions, e.g. {missing[:3]}")

    phi = an._shapley_values(value_fn, blocs)
    vN = value_fn[tuple(sorted(blocs))]

    rows = []
    for b in blocs:
        solo = value_fn[(b,)]
        rows.append({"Bloc": b, "Consumption share (%)": round(100 * share[b], 1),
                     "Shapley allocation ($B)": round(phi[b], 2),
                     "Stand-alone v({i}) ($B)": round(solo, 2),
                     "Individually rational": phi[b] >= max(0.0, solo) - 1e-6})

    all_excess = {S: sum(phi[b] for b in S) - value_fn[S] for S in all_S}
    core_ok = min(all_excess.values()) >= -1e-6
    proper = {S: e for S, e in all_excess.items() if len(S) < len(blocs)}
    worst_S = min(proper, key=proper.get)
    min_proper_excess = proper[worst_S]

    max_gap = max(gaps) if gaps else float("nan")
    obj_uncertainty = max_gap * abs(dt_nc) if gaps else float("nan")
    certified_positive = bool(min_proper_excess > 10.0 * obj_uncertainty) if gaps else False

    df = pd.DataFrame(rows)
    df.to_csv(f"outputs/table_coalition_certify_{tag}.csv", index=False)

    summary = {
        "grand_coalition_value": round(vN, 2),
        "shapley_sum": round(sum(phi.values()), 2),
        "core_stable": bool(core_ok),
        "min_proper_core_excess": round(float(min_proper_excess), 3),
        "closest_blocking_coalition": "+".join(worst_S),
        "mip_gap_target": base["mip_gap_target"],
        "max_achieved_gap": round(float(max_gap), 8),
        "obj_uncertainty_B": round(float(obj_uncertainty), 4),
        "core_excess_certified_positive": certified_positive,
        "n_shards": len(shards),
    }
    with open(f"outputs/core_certify_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote outputs/core_certify_{tag}.json and table_coalition_certify_{tag}.csv", flush=True)


# core-certify: certified core-stability of the cooperative game (single node)
def cmd_core_certify(args):
    """Re-solve the six-bloc cooperative game (64 coalitions) at a tightened relative gap
    (default CORE_MIP_GAP=1e-4), recording each solve's achieved gap as a numerical-
    uncertainty scale on the objective. Reports the minimum proper-coalition core excess
    with that uncertainty so the "strictly inside the core" claim is certified rather than
    asserted at the coarser 1e-3 gap. Runs the linear game by default; PRICE_CURVE=isoelastic
    certifies the constant-elasticity game. Writes outputs/table_coalition_certify_{tag}.csv
    and core_certify_{tag}.json."""
    TC = int(os.environ.get("COALITION_TARGET_COUNT", "30"))
    GAP = float(os.environ.get("CORE_MIP_GAP", "1e-4"))
    TL = int(os.environ.get("CORE_TIME_LIMIT", "1200"))
    CURVE = os.environ.get("PRICE_CURVE", "linear")
    print(f"[config] target_count={TC} mip_gap={GAP} time_limit={TL} price_curve={CURVE}", flush=True)
    os.makedirs("outputs", exist_ok=True)

    cs = an.coalition_cost_sharing(target_count=TC, price_curve=CURVE, mip_gap=GAP, time_limit=TL)
    tag = "linear" if CURVE == "linear" else CURVE
    cs.to_csv(f"outputs/table_coalition_certify_{tag}.csv", index=False)

    keys = ("grand_coalition_value", "shapley_sum", "core_stable", "min_proper_core_excess",
            "closest_blocking_coalition", "mip_gap_target", "max_achieved_gap",
            "obj_uncertainty_B", "core_excess_certified_positive")
    summary = {k: cs.attrs[k] for k in keys if k in cs.attrs}
    with open(f"outputs/core_certify_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote outputs/core_certify_{tag}.json", flush=True)


# robustness: full robustness and sensitivity sweep
def cmd_robustness(args):
    """Reproduce the robustness and sensitivity checks, writing one CSV per check under
    outputs/ (table_cascade_weight_gap.csv, table_escvi_stability.csv,
    table_transition_robustness.csv, table_loss_distribution.csv, table_attrition_ranking.csv,
    and the further _guard checks below), plus coalition_game_summary.json and
    joint_uncertainty_summary.json.

    Fast checks (no optimization):
      cascade_weight_gap        first-order vs Leontief-inverse cascade weights
      escvi_weight_stability    ESCVI ranking under weight perturbation

    Solve-based checks (each re-solves the full-closure Hormuz MIP):
      transition_perturbation_robustness  transition-row perturbation
      loss_distribution_cvar              27-path loss distribution + CVaR
      attrition_ranking_stability         instrument ranking under attrition

    Defaults are memory-lean desktop settings; on the HPC set COALITION_TARGET_COUNT=30 for
    the full 27-path tree, raise JOINT_DRAWS, and widen FIXED_STAGE_COUNTS (e.g. 4,5,6).
    GEO_RESUME=1 skips any check whose CSV already exists."""
    COALITION_TC = int(os.environ.get("COALITION_TARGET_COUNT", "9"))
    JOINT_DRAWS = int(os.environ.get("JOINT_DRAWS", "200"))
    FIXED_STAGE_COUNTS = tuple(int(x) for x in
                               os.environ.get("FIXED_STAGE_COUNTS", "4,5").split(","))
    print(f"[config] COALITION_TC={COALITION_TC}  JOINT_DRAWS={JOINT_DRAWS}  "
          f"FIXED_STAGE_COUNTS={FIXED_STAGE_COUNTS}", flush=True)

    cascade_weight_gap = an.cascade_weight_gap
    escvi_weight_stability = an.escvi_weight_stability
    transition_perturbation_robustness = an.transition_perturbation_robustness
    loss_distribution_cvar = an.loss_distribution_cvar
    attrition_ranking_stability = an.attrition_ranking_stability
    joint_uncertainty_analysis = an.joint_uncertainty_analysis
    duration_fixed_stage_robustness = an.duration_fixed_stage_robustness
    logistics_derating_robustness = an.logistics_derating_robustness

    RESUME = os.environ.get("GEO_RESUME") == "1"

    def _guard(label, fn, path, preview_rows=12):
        """Run one check and write its CSV, swallowing any failure so one heavy check
        cannot abort the sweep. With GEO_RESUME=1 a check whose CSV exists is skipped, so
        a walltime-killed HPC job can be re-submitted and resumes where it stopped."""
        print(f"\n=== {label} ===", flush=True)
        if RESUME and os.path.exists(path):
            print(f"  RESUME: {path} already exists, skipping.", flush=True)
            return None
        try:
            df = fn()
            df.to_csv(path, index=False)
            print(df.head(preview_rows).to_string(index=False), flush=True)
            return df
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIPPED ({type(exc).__name__}): {exc}", flush=True)
            return None

    print("\n=== Cascade weights: first-order vs Leontief inverse ===", flush=True)
    cwg = cascade_weight_gap()
    cwg.to_csv("outputs/table_cascade_weight_gap.csv", index=False)
    print(cwg.to_string(index=False), flush=True)

    print("\n=== ESCVI ranking stability under weight perturbation ===", flush=True)
    esc = escvi_weight_stability()
    esc.to_csv("outputs/table_escvi_stability.csv", index=False)
    print(esc.to_string(index=False), flush=True)

    print("\n=== Transition-row perturbation (full-closure Hormuz) ===", flush=True)
    tp = transition_perturbation_robustness(deltas=(-0.10, 0.0, 0.10))
    tp.to_csv("outputs/table_transition_robustness.csv", index=False)
    print(tp.to_string(index=False), flush=True)

    print("\n=== 27-path loss distribution and CVaR (full-closure Hormuz) ===", flush=True)
    ld = loss_distribution_cvar()
    ld.to_csv("outputs/table_loss_distribution.csv", index=False)
    print(ld.to_string(index=False), flush=True)

    print("\n=== Instrument ranking under attrition dynamics ===", flush=True)
    ar = attrition_ranking_stability()
    ar.to_csv("outputs/table_attrition_ranking.csv", index=False)
    print(ar.to_string(index=False), flush=True)

    # Additional checks, fast ones first.
    _guard("Import dependence and exposure disclosure",
           an.exposure_disclosure, "outputs/table_exposure_disclosure.csv")

    _guard("Normal-state material-balance reconciliation",
           an.normal_state_reconciliation, "outputs/table_normal_state_reconciliation.csv")

    _guard("PWL approximation error (amplification and deadweight)",
           an.amplification_pwl_error, "outputs/table_pwl_error.csv")

    _guard("Cascade four-cell cross-evaluation",
           an.cascade_cross_evaluation, "outputs/table_cascade_cross_eval.csv")

    _guard("Duration under a fixed decision clock",
           lambda: duration_fixed_stage_robustness(stage_counts=FIXED_STAGE_COUNTS),
           "outputs/table_duration_fixed_stage.csv")

    _guard("Logistics realism: derated delivery capacity",
           logistics_derating_robustness, "outputs/table_logistics_derating.csv")

    # Coordination layer: noncooperative vs planner, and cooperative cost-sharing.
    _guard("Coordination analysis (noncooperative vs planner, monopsony)",
           an.coordination_analysis, "outputs/table_coordination_analysis.csv")

    cs = _guard("Coalition cost-sharing across regional blocs (Shapley, IR, core)",
                lambda: an.coalition_cost_sharing(target_count=COALITION_TC),
                "outputs/table_coalition_cost_sharing.csv")
    if cs is not None:
        with open("outputs/coalition_game_summary.json", "w") as fh:
            json.dump({k: cs.attrs[k] for k in ("grand_coalition_value", "shapley_sum",
                                                 "core_stable", "min_proper_core_excess",
                                                 "closest_blocking_coalition")
                       if k in cs.attrs}, fh, indent=2)

    # Heaviest check last: Monte-Carlo joint uncertainty. Default JOINT_DRAWS=200 matches
    # the paper; lower it for a faster approximate pass.
    def _joint():
        return joint_uncertainty_analysis(n_draws=JOINT_DRAWS)
    ju = _guard(f"Joint uncertainty (Monte-Carlo, {JOINT_DRAWS} draws)", _joint,
                "outputs/table_joint_uncertainty.csv")
    if ju is not None:
        with open("outputs/joint_uncertainty_summary.json", "w") as fh:
            json.dump({k: ju.attrs[k] for k in ("summary", "policy_frequency",
                                                 "n_draws", "n_skipped")
                       if k in ju.attrs}, fh, indent=2)

    print("\nDONE. Wrote the robustness CSVs in outputs/.", flush=True)


# build-cascade: rebuild the six-sector cascade matrix beta from WIOD tables
def cmd_build_cascade(args):
    """Reproducibly build the six-sector cascade matrix beta (config.BETA and
    data/cascade_beta.csv) from the WIOD 2016 world input-output tables (Timmer et al.,
    2015; Groningen Dataverse DOI 10.34894/PJ2M1C, WIOTS_in_EXCEL release, one .xlsb per year).

    Energy-restricted method: the six sectors are energy-demand categories, so beta carries
    each sector's dependence on ENERGY inputs, not on all intermediate inputs by dollar value.
    The supplier side is restricted to the energy-carrying WIOD industries (D35 -> electricity,
    C19 -> industry, H49..H53 -> transport). Per country, beta[a, s] is the share of user
    sector a's total intermediate input that is energy of type s (diagonal excluded), averaged
    over the major economies and rescaled by one global factor to TARGET_INTENSITY. Writes
    data/cascade_beta.csv and prints the config.BETA literal."""
    from pathlib import Path
    import numpy as np  # noqa: F401
    import pandas as pd

    DATA = Path(HERE).resolve().parent / "data"
    WIOD_DIR = DATA / "WIOTS_in_EXCEL"

    SECTORS = ["electricity", "industry", "agriculture",
               "transport", "residential", "healthcare"]

    # USER-side concordance: WIOD (ISIC Rev. 4, A*56) industry codes -> model sectors.
    WIOD_TO_SECTOR = {
        "D35": "electricity",
        "B": "industry", "C10-C12": "industry", "C13-C15": "industry",
        "C16": "industry", "C17": "industry", "C19": "industry",
        "C20": "industry", "C22": "industry", "C23": "industry",
        "C24": "industry", "C25": "industry", "C26": "industry",
        "C28": "industry", "C29": "industry", "F": "industry",
        "E36": "industry", "E37-E39": "industry",
        "A01": "agriculture", "A02": "agriculture", "A03": "agriculture",
        "H49": "transport", "H50": "transport", "H51": "transport",
        "H52": "transport", "H53": "transport",
        "L68": "residential", "T": "residential",
        "Q": "healthcare", "O84": "healthcare", "P85": "healthcare",
    }

    # SUPPLIER-side energy industries -> model sector (the only nonzero beta columns).
    ENERGY_SUP = {"D35": "electricity", "C19": "industry",
                  "H49": "transport", "H50": "transport", "H51": "transport",
                  "H52": "transport", "H53": "transport"}

    # Economies averaged over (major crude oil importers present in WIOD 2016).
    TARGET = ["CHN", "IND", "JPN", "KOR", "TWN", "IDN", "TUR",
              "USA", "DEU", "FRA", "ITA", "ESP", "GBR", "NLD", "BRA"]
    YEARS = [2014]
    TARGET_INTENSITY = 2.58   # calibrated overall cascade intensity (sum of beta)

    def _read_year(year):
        """Read one WIOT .xlsb file, returning the TARGET-economy rows plus the column
        bookkeeping needed to slice each one."""
        f = WIOD_DIR / f"WIOT{year}_Nov16_ROW.xlsb"
        hdr = pd.read_excel(f, engine="pyxlsb", header=None, nrows=6)
        use_ind = hdr.iloc[2].astype("string").str.strip()     # using-industry code per column
        use_ctry = hdr.iloc[4].astype("string").str.strip()    # using-country per column

        present, positions = [], {}
        for cc in TARGET:
            mask = use_ctry.eq(cc).fillna(False).to_numpy()
            pos = [p for p in range(len(mask)) if mask[p]]
            if pos:
                present.append(cc)
                positions[cc] = pos

        all_pos = sorted(set([0, 1, 2, 3] + [p for cc in present for p in positions[cc]]))
        pos_to_local = {p: i for i, p in enumerate(all_pos)}
        raw = pd.read_excel(f, engine="pyxlsb", header=None, skiprows=6, usecols=all_pos)
        raw.columns = range(raw.shape[1])
        return raw, present, positions, pos_to_local, use_ind

    def _country_beta(raw, positions, pos_to_local, use_ind, cc):
        """Energy-restricted 6x6 beta for one country from the read frame."""
        ccol = raw.iloc[:, 2].astype("string").str.strip()
        sub = raw.loc[ccol.eq(cc).fillna(False).to_numpy()]
        local = [pos_to_local[p] for p in positions[cc]]
        codes = [use_ind.iloc[p] for p in positions[cc]]

        Z = sub.iloc[:, local].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        Z.index = sub.iloc[:, 0].astype("string").str.strip().to_numpy()   # supplier code
        Z.columns = codes
        Z = Z.loc[:, [c for c in Z.columns if c in set(Z.index)]]          # 56x56 intermediate

        col_sec = pd.Series({c: WIOD_TO_SECTOR.get(c) for c in Z.columns})
        Zc = Z.T.groupby(col_sec).sum().T                                  # 56 suppliers x 6 users
        x = Zc.sum(axis=0)                                                 # total input per user sector

        erows = Zc.loc[[c for c in Zc.index if c in ENERGY_SUP]]
        row_sec = pd.Series({c: ENERGY_SUP[c] for c in erows.index})
        G = erows.groupby(row_sec).sum()                                  # energy sectors x 6 users
        A = G.div(x, axis=1).fillna(0.0)                                   # energy-s share of a's inputs
        beta = A.T.reindex(index=SECTORS, columns=SECTORS).fillna(0.0)
        for s in SECTORS:
            beta.loc[s, s] = 0.0
        return beta

    def build(years=YEARS, intensity=TARGET_INTENSITY):
        acc, n = None, 0
        for yr in years:
            raw, present, positions, pos_to_local, use_ind = _read_year(yr)
            for cc in present:
                b = _country_beta(raw, positions, pos_to_local, use_ind, cc)
                acc = b if acc is None else acc + b
                n += 1
            print(f"{yr}: averaged {len(present)} economies")
        mean = acc / n
        beta = (mean * (intensity / mean.values.sum())).round(2)          # rescale to target intensity
        for s in SECTORS:
            beta.loc[s, s] = 0.0
        beta.index.name = "sector"
        return beta

    beta = build()
    beta.to_csv(DATA / "cascade_beta.csv")
    print(f"\nWrote {DATA / 'cascade_beta.csv'}  (total intensity {beta.values.sum():.3f})")
    print(beta.to_string())
    print("\n# config.BETA literal:")
    for a in SECTORS:
        items = ", ".join(f'"{s}": {beta.loc[a, s]:.2f}'
                          for s in SECTORS if beta.loc[a, s] >= 0.005)
        print(f'    "{a}":{" " * (13 - len(a))}{{{items}}},')


# CLI
SUBCOMMANDS = [
    ("reconcile", cmd_reconcile,
     "three-regime reconciliation + computational disclosure"),
    ("coalition", cmd_coalition,
     "coordination re-solve, physical clearing and normalized cooperative game"),
    ("ampspec", cmd_ampspec,
     "amplification-specification benchmark (headline/capped/leontief)"),
    ("displacement", cmd_displacement,
     "reserve/substitute market-displacement sensitivity (phi=0/0.5/1)"),
    ("isoelastic", cmd_isoelastic,
     "constant-elasticity inverse-demand three-regime benchmark"),
    ("isogame", cmd_isogame,
     "constant-elasticity cooperative game and core"),
    ("grid-refine", cmd_grid_refine,
     "deadweight-grid refinement (12 vs 24 breakpoints) for the isoelastic benchmark"),
    ("stagelen", cmd_stagelen,
     "stage-length (decision-frequency) sensitivity of the headline result"),
    ("core-shard", cmd_core_shard,
     "one shard of the certified core-stability re-run (HPC fan-out)"),
    ("core-combine", cmd_core_combine,
     "combine the core-certification shards into the final certified game"),
    ("core-certify", cmd_core_certify,
     "certified core-stability of the cooperative game (single node)"),
    ("robustness", cmd_robustness,
     "full robustness and sensitivity sweep, writing CSVs"),
    ("build-cascade", cmd_build_cascade,
     "rebuild the six-sector cascade matrix beta from the WIOD tables"),
]


def build_parser():
    parser = argparse.ArgumentParser(
        prog="experiments.py",
        description="Consolidated driver for the paper's re-run and robustness "
                    "experiments. Configuration is read from environment variables "
                    "(preserved for the HPC job scripts); see each subcommand's help.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True,
                                metavar="<subcommand>")
    for name, fn, help_text in SUBCOMMANDS:
        p = sub.add_parser(name, help=help_text,
                           description=(fn.__doc__ or help_text),
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        p.set_defaults(func=fn)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
