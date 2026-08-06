# Data provenance and calibration sources

This file documents where every calibration input in the model comes from and
how it should be interpreted. Inputs fall into three classes:

- **Public data**, taken from a named public statistical source.
- **Constructed**, assembled or proxied from public data through a stated rule
  (for example, regional sector shares carved from consumption balances).
- **Modeling assumption**, a structural choice or illustrative order of
  magnitude, not a measured quantity. These are stated as such and probed in the
  sensitivity analysis rather than presented as estimates.

All numeric values live in `code/model.py` (the `CONFIG` block and the
`build_countries`, `build_sources`, `build_chokepoints`, and `build_processing`
functions). The 53-country table is also exported for inspection to
`outputs/data_countries.csv`.

## Country-level inputs (53 importers)

| Field | Source | Class |
|---|---|---|
| Oil demand (mbd) | EIA / IEA country oil balances | Public data |
| Chokepoint dependence (Hormuz, Malacca, Suez share of supply) | EIA *World Oil Transit Chokepoints* assessment, apportioned to importers by trade direction | Constructed from public data |
| Strategic reserve coverage (days) | IEA 90-day stockholding obligation for members, low-coverage buffers for specific non-members | Public data (members) / Constructed (non-members) |
| GDP (USD billion) | World Bank national accounts | Public data |
| Population (millions) | World Bank | Public data |
| Renewable share (%) | IEA / IRENA | Public data |
| End-use sector shares (electricity, industry, agriculture, transport, residential, healthcare) | IEA oil final-consumption balances, as regional averages | Constructed from public data |
| "Healthcare" sector share | Proxy for critical-services oil use (hospitals, water, cold chain, emergency logistics), carved from the residential/commercial and transport shares | Constructed proxy |
| Max reserve draw rate (10% of daily demand) | n/a | Modeling assumption |

The chokepoint-dependence fractions and the low-coverage reserve buffers are
apportioned from aggregate flows and country balances rather than measured
per-country, so the paper treats them as constructed inputs and reports that
moderate perturbations leave the country ranking nearly fixed (see
`analysis.py` and the sensitivity tables).

## Price-channel parameters (`CONFIG` block)

| Constant | Value | Source | Class |
|---|---|---|---|
| Baseline oil price `P0` | 72 USD/bbl (Brent) | EIA / IEA | Public data |
| Short-run demand elasticity `eta_D` | -0.20 (elastic end of the empirical range) | Caldara, Cavallo, Iacoviello (2019) | Public data |
| World liquids supply `Qbar` | 102 mbd | EIA / IEA | Public data |
| Spare capacity reaching the market without crossing the chokepoint | ~2 mbd | EIA 2024-25 (world surplus ~4 mbd, most behind Hormuz) | Constructed from public data |

## Source regions and chokepoints

| Input | Source | Class |
|---|---|---|
| Source-region export capacities (mbd) | EIA / OPEC crude production statistics | Public data |
| Source-region chokepoint-routed fractions | EIA *World Oil Transit Chokepoints* assessment | Public data |
| Chokepoint normal throughput (mbd) | EIA *World Oil Transit Chokepoints* assessment | Public data |
| Bypass pipeline capacities (Habshan-Fujairah, Saudi East-West) | Public pipeline capacity figures | Public data |
| Processing / refinery hub capacities | EIA / IEA refinery statistics | Public data |

## Cross-sector cascade matrix

| Input | Source | Class |
|---|---|---|
| Inter-sector dependence coefficients `beta_ss'` (`data/cascade_beta.csv`) | Derived from the World Input-Output Database (WIOD) 2016 release, then rescaled to the calibrated overall intensity | Constructed from public data |
| Cascade regional intensity scalars | Calibrated to the target spectral radius | Modeling assumption |

`data/cascade_beta.csv` is regenerated from the WIOD source tables by
`code/experiments.py build-cascade`. The WIOD tables themselves are not redistributed
here (see `data/README.md`).

## Cost and policy parameters (`CONFIG` block)

| Input | Value | Class |
|---|---|---|
| Direct-shortage penalty `pi_ns` | End-use value of a delivered barrel, ~150 USD/bbl scale, by sector | Constructed from public data |
| Emergency substitute channel costs (product-stock 70, refinery-yield 95, fuel-switch 120 USD/bbl) | Ordered second-best resource costs | Modeling assumption |
| Substitute total capability (5% of demand per stage) and channel split (0.40 / 0.35 / 0.25) | Illustrative capability, uniform across importers | Modeling assumption |
| Product-stock inventory (50 days of daily release capacity) | Fixed physical barrel stock | Modeling assumption |
| Policy activation costs `gamma_p` | Order-of-magnitude fixed charges below the damages they avert | Modeling assumption |
| Sector damage-function parameters (threshold, curvature) | Paper appendix calibration | Modeling assumption |

The substitute capabilities and activation costs are deliberately illustrative.
Their influence on the results is bounded in the joint-uncertainty sweep (total
capability 3-8% of demand, per-barrel costs +/-20%) and the policy-cost
robustness check.
