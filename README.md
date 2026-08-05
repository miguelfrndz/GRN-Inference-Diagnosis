# GRN Inference Diagnosis

Code for the paper **"When Does Gene Regulatory Network Inference Break? A Controlled Diagnostic Study of Causal and Correlational Methods on Single-Cell Data"** (preprint available at [arXiv:2605.04930](https://arxiv.org/abs/2605.04930)).

The repository compares representative gene regulatory network (GRN) inference methods under controlled single-cell pathologies: dropout, latent confounding, cell-type mixing, feedback, graph density, sample size, and pseudotime drift.

## Repository Layout

- `src/simulator.py`: synthetic GRN simulator with linear and nonlinear SCMs.
- `src/methods.py`: inference methods used in the benchmark.
- `src/metrics.py`: AUPRC and error decomposition metrics.
- `src/experiments.py`: experiment grids and runners.
- `src/plotting.py`: figure and table generation.
- `src/run_all.py`: command-line entry point.

Generated files are written to:

- `results/`: experiment CSVs.
- `figures/`: paper figures in PNG and PDF.
- `tables/`: summary CSV tables.

These output directories are ignored by git so runs can be regenerated locally.

## Installation

This project uses Python 3.13 and `uv`.

```bash
uv sync
```

You can also run the code with any Python 3.13 environment that has the dependencies listed in `pyproject.toml`.

## Quick Start

Run a small linear-SCM sweep and build figures/tables:

```bash
uv run python -m src.run_all --quick
```

Run the full default linear-SCM sweep:

```bash
uv run python -m src.run_all
```

The default command writes `results/results.csv`, then builds the standard figures and tables.

Run this linear sweep before running optional sweeps by themselves. The plotting code uses `results/results.csv` as the baseline for shared figures.

## Experiment Guide

### Linear Synthetic Sweep

This is the main synthetic benchmark. It sweeps each pathology independently over five levels and evaluates all methods over multiple random seeds.

```bash
uv run python -m src.run_all --n-seeds 10
```

Use fewer seeds for iteration:

```bash
uv run python -m src.run_all --quick
```

### Nonlinear Synthetic Sweep

Run the same pathology grid with the nonlinear `tanh` SCM:

```bash
uv run python -m src.run_all --nonlinear --n-seeds 10
```

This expects the linear results to exist because the plotting step overlays nonlinear results against the linear baseline. To run both in one command, use `--all` or run the linear sweep first.

### Interaction Sweep

Run the joint dropout x confounders x density sweep:

```bash
uv run python -m src.run_all --interaction --interaction-seeds 5
```

The interaction results are saved to `results/results_interaction.csv`.

As above, run the linear sweep first if `results/results.csv` does not already exist.

### All Experiments

Run the linear, nonlinear, and interaction experiments:

```bash
uv run python -m src.run_all --all
```

### Rebuild Figures Only

If the result CSVs already exist, regenerate figures and tables without recomputing experiments:

```bash
uv run python -m src.run_all --figures-only
uv run python -m src.run_all --nonlinear --figures-only
uv run python -m src.run_all --interaction --figures-only
uv run python -m src.run_all --all --figures-only
```

## Methods

The benchmark includes:

- Pearson correlation.
- Mutual information.
- GENIE3-style random forest feature importance.
- PC-style conditional independence testing.
- GES-style greedy BIC search.
- NOTEARS.

Undirected and directed AUPRC are reported for every method. Error decomposition is computed at a top-K threshold where K is the number of true directed edges.

## Citation

If you use this code in your research, please cite:
```bibtex
@article{fernandez2026causalGRN,
    title={When Does Gene Regulatory Network Inference Break? A Controlled Diagnostic Study of Causal and Correlational Methods on Single-Cell Data},
    author={Fernandez-de-Retana, Miguel and Sanchez-Corcuera, Ruben and Zulaika, Unai and Bilbao-Jayo, Aritz and Almeida, Aitor},
    year={2026}
    journal={arXiv preprint arXiv:2605.04930},
}
```

## License

This code is released under the MIT License.
