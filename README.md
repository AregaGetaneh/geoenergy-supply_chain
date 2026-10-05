# Crude oil supply at the refinery gate during maritime disruptions

Code and data that reproduce the ten published figures and the seven supplementary
tables of the accompanying paper.

## Running it

```
conda env create -f environment.yml
conda activate crude-disruption
python run_all.py --check
```

`run_all.py` draws every figure into `outputs/` and writes the supplementary tables into
`outputs/tables/`. The `--check` flag then compares each drawn figure with the published
version in `reference/` and prints a pass or fail line for each.

A run takes about two minutes. Everything runs offline once the environment is built,
with one exception noted under *Maps* below.

## What the check verifies

For each of the ten figures it compares the page size, the axes bounding boxes of every
panel, the full text content, and the vector drawing content. Byte equality is not
required and would not be meaningful: a PDF records its creation time, so two identical
runs never produce identical bytes.

Whitespace is normalised before the text comparison. Mathtext spacing around `=` differs
between matplotlib versions, which is kerning rather than content.

## Layout

```
run_all.py            entry point
src/style.py          palette, type sizes, axis labels, panel titles, save helper
src/figures.py        the ten published figures, one function per plate
src/verify.py         the comparison described above
data/jodi/            the JODI-Oil crude extract, with its provenance record
data/panel/           the derived tables the figures read
data/market/          transits, freight, and price series
data/DIGESTS.md       SHA-256 of every data file
reference/            the published figures, for comparison
outputs/              everything the run produces
```

## Data

| Source | What is here | Terms |
|---|---|---|
| JODI-Oil World Database | derived crude extract, 2020 to 2026, with per-file digests of the originals | see `data/jodi/SOURCE_PROVENANCE.txt` |
| IMF PortWatch / UN Global Platform | monthly tanker transits for six chokepoints | source URLs in `data/market/` |
| Eurostat Comext | crude import mode split, derived | reuse with attribution |
| US EIA | Brent spot, monthly average and month end | US federal data |
| Baltic Exchange | TD3C monthly means of weekly published values, with methodology URL | aggregated, see `scope_note` |

The ICE settlement series used to form the price difference is **not redistributed here**.
`data/market/brent_price_curve.csv` carries the EIA columns and the derived difference
only. The original is available from ICE.

Natural Earth map geometry is not bundled. Cartopy downloads it on first use.

### Maps

Three figures draw coastlines and borders through cartopy, which fetches Natural Earth
files the first time it needs them. That one step requires a network connection. Every
later run, and every other figure, works offline.

## Figures

| Figure | File |
|---|---|
| 1 | `Fig1_gulf_anatomy` |
| 2 | `Fig6_transport_boundary` |
| 3 | `Fig2_world_outcomes` |
| 4 | `Fig3_absorption` |
| 5 | `Fig5_inventory_buffering` |
| 6 | `Fig8_boundaries` |
| 7 | `Fig7_episode_maps` |
| 8 | `Fig4_arrivals_vs_availability` |
| S1 | `FigS1_closure_residuals` |
| S5 | `FigS5_corridor_dependence` |

The file names follow the order in which the figures were built, not the order in which
the paper presents them. The table above maps one to the other.

## Licence

Code is MIT, see `LICENSE`. Data files carry the terms of their original providers.
