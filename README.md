# GeoEnergy-SCRO

**Stochastic Chokepoint Resilience Optimizer**, a multi-stage stochastic
mixed-integer optimization (MIP) framework for maritime crude oil chokepoint
disruption policy design.

All quantitative results come from the stochastic MIP. The headline benchmark is
a **100-day** Strait of Hormuz disruption modeled as **four 25-day decision
stages** over a 27-path (3^3) scenario tree, sensitivity experiments sweep other
horizons and other chokepoints (Malacca, Suez).

This repository accompanies the paper by Abate, Keles, Zhang, and Liu
(DTU Management). See [Citation](#citation) below.

## Project layout

```text
geoenergy-supply_chain-mip-v4/
├── code/
│   ├── model.py            # config, network data, stochastic MIP, solve wrapper
│   ├── analysis.py         # result tables, figures, cooperative game, robustness checks
│   ├── experiments.py      # CLI for every targeted experiment (see below)
│   ├── pipeline.py         # end-to-end reproduction (headline tables + all figures)
│   ├── submit.sh           # DTU HPC (LSF) job script, parameterized by experiment
│   └── requirements.txt    # Python dependencies
├── data/       # cascade_beta.csv (+ WIOD source tables, downloaded separately, see data/README.md)
├── DATA_PROVENANCE.md   # source and interpretation of every calibration input
├── LICENSE              # MIT
└── README.md
```

`model.py` builds and solves the optimization. `analysis.py` turns solved models
into the paper's tables, figures, cooperative-game results, and robustness
checks. `pipeline.py` runs the whole study end to end, and `experiments.py`
reproduces individual experiments through subcommands.

## Quick start

### 1. Environment

Python >= 3.10 with `numpy`, `scipy`, `pandas`, `matplotlib`, `pyomo`, `cartopy`,
and a MIP solver. The paper's numbers are produced with **Gurobi** (`gurobipy`,
academic licenses are free), **HiGHS** (`highspy`) is an open-source fallback and
is selected automatically when Gurobi is not licensed.

```bash
pip install -r code/requirements.txt
```

### 2. Run the pipeline

Run the full study headless from the repository root:

```bash
python code/pipeline.py
```

Each run re-solves every MIP from scratch and writes all tables and figures into
a local `outputs/` directory. A full run takes roughly an hour (mostly the
Shapley attribution and the efficiency-equity frontiers). Solved MIPs are cached
under `outputs/mip_cache/`, so a rerun reuses completed solves.

## Reproducing specific results

`pipeline.py` reproduces the headline tables and all figures. Individual
experiments are reproduced through subcommands of `experiments.py`, run from the
repository root, for example `python code/experiments.py reconcile`. Each writes
named files into `outputs/`.

| Command | Result |
|---|---|
| `python code/pipeline.py` | Headline saving, instrument attribution, exposure/loss maps, flow Sankey, scenario bars, timing and frontier figures |
| `experiments.py reconcile` | Three-regime welfare decomposition and the solve-size disclosure |
| `experiments.py coalition` | Cooperative game: Shapley allocation, core, and transfers (realized incidence) |
| `experiments.py ampspec` | Amplification specifications (headline vs capped vs Leontief cascade) |
| `experiments.py displacement` | Saving under zero / partial / one-for-one market displacement |
| `experiments.py isoelastic` | Constant-elasticity price benchmark (three regimes) |
| `experiments.py isogame` | Cooperative game under constant-elasticity demand |
| `experiments.py grid-refine` | Deadweight tangent-grid refinement (12 vs 24 breakpoints) |
| `experiments.py stagelen` | Stage-length (decision-frequency) robustness |
| `experiments.py core-shard` then `core-combine` | Certified core stability at a tightened optimality gap (HPC fan-out) |
| `experiments.py core-certify` | Certified core stability in a single job |
| `experiments.py robustness` | Cascade-weight, transition, CVaR, and attrition checks |
| `experiments.py build-cascade` | Rebuild `data/cascade_beta.csv` from the WIOD tables |

Most subcommands read environment variables to select the scenario count, gap,
or price curve, run `python code/experiments.py <name> --help` or see the
docstrings in `experiments.py`.

### Reproducing the full manuscript

The paper's numbers use the full 27-path (3^3) scenario tree solved with Gurobi
at a fixed seed of 42. Running the pipeline and then each experiment command
regenerates every table and figure in the manuscript:

```bash
python code/pipeline.py                                           # headline tables, policy attribution over the 64 instrument subsets, all figures
python code/experiments.py reconcile                              # three-regime reconciliation and the solve-size table
COALITION_TARGET_COUNT=30 python code/experiments.py coalition    # 64-coalition cooperative game, realized incidence
COALITION_TARGET_COUNT=30 python code/experiments.py core-certify # certified core stability at the 1e-4 gap
python code/experiments.py ampspec                                # amplification specifications
python code/experiments.py isoelastic                             # constant-elasticity benchmark
COALITION_TARGET_COUNT=30 python code/experiments.py isogame      # constant-elasticity cooperative game
python code/experiments.py displacement                           # market-displacement sensitivity
python code/experiments.py robustness                             # 200-draw joint-uncertainty sweep and the sensitivity checks
```

Gurobi produces the paper's numbers. HiGHS reproduces the same model with a
different solver and is selected automatically when Gurobi is not licensed.

## Data

The main pipeline needs only `data/cascade_beta.csv`, which is provided. The WIOD
2016 source tables used to derive it are large and licensed by their publisher,
so they are downloaded separately, see **`data/README.md`**. The source and
interpretation of every calibration input (country demand, chokepoint
dependence, reserve coverage, elasticity, cascade matrix, and cost parameters)
are documented in **`DATA_PROVENANCE.md`**.

## Notes

- The pipeline is deterministic. The same code and solver reproduce the same
  numbers.
- Running the code creates a local `outputs/` directory holding every table and
  figure. It is not part of this repository.
- `outputs/mip_cache/` holds solved MIPs to warm-start reruns.

## License

Source code is released under the MIT License (see `LICENSE`). The WIOD input
tables and the public statistical sources behind the calibration retain their
own terms, see `data/README.md` and `DATA_PROVENANCE.md`.

## Citation

If you use this code, please cite the accompanying paper.

> Abate, A. G., Keles, D., Zhang, L., Liu, J. Coordinated response to maritime
> crude oil chokepoint disruptions. A multi-stage stochastic optimization approach.
> DTU Management. (Under review.)

A BibTeX entry will be added on publication.
