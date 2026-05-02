"""
Each pathology is swept over a range of intensities with multiple seeds per
level. For each (pathology, level, seed, method), the output records:
    - AUPRC (undirected skeleton)
    - AUPRC (directed)
    - error-type decomposition counts
    - runtime (seconds)

Rows are written incrementally so long sweeps can be resumed.
"""

from __future__ import annotations

import time
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator import simulate, SimConfig
from src.methods import METHODS, METHOD_CLASS
from src.metrics import auprc_undirected, auprc_directed, error_decomposition

# Pathology sweeps
SWEEPS = {
    "dropout": {
        "levels":       [0.0, 0.2, 0.4, 0.6, 0.8],
        "fixed":        dict(n_genes=25, n_cells=800, density=0.1),
        "description":  "Technical zero-inflation (missingness)",
    },
    "n_confounders": {
        "levels":       [0, 2, 4, 8, 16],
        "fixed":        dict(n_genes=25, n_cells=800, density=0.1),
        "description":  "Unobserved latent variables driving gene groups",
    },
    "mix_ratio": {
        "levels":       [0.0, 0.1, 0.25, 0.4, 0.5],
        "fixed":        dict(n_genes=25, n_cells=800, density=0.1),
        "description":  "Cell-type mixing (samples from 2 distinct SCMs)",
    },
    "feedback": {
        "levels":       [0.0, 0.1, 0.2, 0.3, 0.5],
        "fixed":        dict(n_genes=25, n_cells=800, density=0.1),
        "description":  "Probability of adding a back-edge (cycles)",
    },
    "density": {
        "levels":       [0.05, 0.1, 0.15, 0.2, 0.3],
        "fixed":        dict(n_genes=25, n_cells=800),
        "description":  "Edge density of the ground-truth DAG",
    },
    "n_cells": {
        "levels":       [200, 400, 800, 1600, 3200],
        "fixed":        dict(n_genes=25, density=0.1),
        "description":  "Number of cells (sample size)",
    },
    "pseudotime_drift": {
        "levels":       [0.0, 0.2, 0.5, 1.0, 1.5],
        "fixed":        dict(n_genes=25, n_cells=800, density=0.1),
        "description":  "Non-stationarity along pseudotime",
    },
}

DIRECTED_METHODS = {"GENIE3", "PC", "GES", "NOTEARS"}

# Interaction sweep
INTERACTION_GRID = {
    "dropout":       [0.0, 0.3, 0.6, 0.8],
    "n_confounders": [0,   2,   8,   16],
    "density":       [0.05, 0.1, 0.2, 0.3],
}
INTERACTION_FIXED = dict(n_genes=25, n_cells=800)


# Single experiment runner
def run_one(pathology: str, level, seed: int, scm_type: str = "linear") -> list[dict]:
    """Run all methods once at a given pathology level and seed.

    Returns a list of dicts (one per method) ready for DataFrame ingestion.
    """
    base = dict(SWEEPS[pathology]["fixed"])
    base[pathology] = level
    base["seed"] = seed
    base["scm_type"] = scm_type

    out = simulate(SimConfig(**base))

    rows = []
    for name, fn in METHODS.items():
        t0 = time.time()
        try:
            S = fn(out.X)
        except Exception as e:
            print(f"  [fail] {name}: {e}")
            continue
        dt = time.time() - t0
        directed = name in DIRECTED_METHODS
        au_und = auprc_undirected(S, out.A_true)
        au_dir = auprc_directed(S, out.A_true)
        ed = error_decomposition(S, out.A_true, directed=directed)
        rows.append({
            "pathology":    pathology,
            "level":        level,
            "seed":         seed,
            "method":       name,
            "method_class": METHOD_CLASS[name],
            "scm_type":     scm_type,
            "auprc_und":    au_und,
            "auprc_dir":    au_dir,
            "runtime_s":    dt,
            **{f"err_{k}": v for k, v in ed.items()},
        })
    return rows


# Run the full sweep
def run_all_sweeps(
    n_seeds: int = 10,
    out_path: str = "results/results.csv",
    verbose: bool = True,
    resume: bool = True,
    scm_type: str = "linear",
) -> pd.DataFrame:
    """Run the full pathology x level x seed x method grid.

    Saves incrementally after every cell so a crash or ^C does not lose work.
    If resume=True and the output CSV already exists, skips already-computed
    (pathology, level, seed) triples.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing rows for resume support.
    done_keys: set[tuple] = set()
    if resume and out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            for _, r in prev.iterrows():
                done_keys.add((r["pathology"], r["level"], int(r["seed"]), r["method"]))
            if verbose:
                print(f"[resume] loaded {len(prev)} existing rows from {out_path}")
        except Exception as e:
            if verbose:
                print(f"[resume] could not load existing CSV: {e}")

    # Header on first write if file does not exist.
    header_needed = not out_path.exists()

    total = sum(len(s["levels"]) * n_seeds for s in SWEEPS.values())
    done = 0
    all_rows: list[dict] = []
    for pathology, spec in SWEEPS.items():
        for level in spec["levels"]:
            for seed in range(n_seeds):
                done += 1
                # Skip completed cells when every method has already run.
                expected = {m for m in METHODS}
                already = {m for (p, l, s, m) in done_keys
                           if p == pathology and l == level and s == seed}
                if expected.issubset(already):
                    if verbose:
                        print(f"[{done:3d}/{total}] {pathology}={level} seed={seed} (cached)")
                    continue
                if verbose:
                    print(f"[{done:3d}/{total}] {pathology}={level} seed={seed}")
                rows = run_one(pathology, level, seed, scm_type=scm_type)
                all_rows.extend(rows)
                # Save incrementally after every cell in case of crash or ^C.
                pd.DataFrame(rows).to_csv(
                    out_path, mode="a", index=False, header=header_needed,
                )
                header_needed = False
    # Reload the full CSV
    df = pd.read_csv(out_path)
    if verbose:
        print(f"\nTotal rows in {out_path}: {len(df)}")
    return df

# Interaction sweep runner
def run_one_interaction(dropout: float, n_confounders: int, density: float,
                        seed: int, scm_type: str = "linear") -> list[dict]:
    """Run all methods at one (dropout, n_confounders, density, seed) point."""
    cfg = dict(INTERACTION_FIXED)
    cfg.update(dropout=dropout, n_confounders=n_confounders, density=density,
               seed=seed, scm_type=scm_type)
    out = simulate(SimConfig(**cfg))

    rows = []
    for name, fn in METHODS.items():
        t0 = time.time()
        try:
            S = fn(out.X)
        except Exception as e:
            print(f"  [fail] {name}: {e}")
            continue
        dt = time.time() - t0
        directed = name in DIRECTED_METHODS
        au_und = auprc_undirected(S, out.A_true)
        au_dir = auprc_directed(S, out.A_true)
        ed = error_decomposition(S, out.A_true, directed=directed)
        rows.append({
            "dropout":       dropout,
            "n_confounders": n_confounders,
            "density":       density,
            "seed":          seed,
            "method":        name,
            "method_class":  METHOD_CLASS[name],
            "scm_type":      scm_type,
            "auprc_und":     au_und,
            "auprc_dir":     au_dir,
            "runtime_s":     dt,
            **{f"err_{k}": v for k, v in ed.items()},
        })
    return rows

def run_interaction_sweep(
    n_seeds: int = 5,
    out_path: str = "results/results_interaction.csv",
    verbose: bool = True,
    resume: bool = True,
    scm_type: str = "linear",
) -> pd.DataFrame:
    """Joint sweep over the dropout x n_confounders x density grid.

    Records to CSV incrementally with resume support, identical to run_all_sweeps.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys: set[tuple] = set()
    if resume and out_path.exists():
        try:
            prev = pd.read_csv(out_path)
            for _, r in prev.iterrows():
                done_keys.add((float(r["dropout"]), int(r["n_confounders"]),
                               float(r["density"]), int(r["seed"]), r["method"]))
            if verbose:
                print(f"[resume] loaded {len(prev)} existing rows from {out_path}")
        except Exception as e:
            if verbose:
                print(f"[resume] could not load existing CSV: {e}")

    header_needed = not out_path.exists()

    drops = INTERACTION_GRID["dropout"]
    confs = INTERACTION_GRID["n_confounders"]
    dens  = INTERACTION_GRID["density"]
    total = len(drops) * len(confs) * len(dens) * n_seeds
    done = 0

    for d, c, rho in itertools.product(drops, confs, dens):
        for seed in range(n_seeds):
            done += 1
            expected = set(METHODS)
            already = {m for (dd, cc, rr, ss, m) in done_keys
                       if dd == d and cc == c and rr == rho and ss == seed}
            if expected.issubset(already):
                if verbose:
                    print(f"[{done:4d}/{total}] d={d} c={c} rho={rho} seed={seed} (cached)")
                continue
            if verbose:
                print(f"[{done:4d}/{total}] d={d} c={c} rho={rho} seed={seed}")
            rows = run_one_interaction(d, c, rho, seed, scm_type=scm_type)
            pd.DataFrame(rows).to_csv(out_path, mode="a", index=False, header=header_needed)
            header_needed = False
    df = pd.read_csv(out_path)
    if verbose:
        print(f"\nTotal rows in {out_path}: {len(df)}")
    return df
