"""
Run experiment sweeps and regenerate figures/tables.

Usage:
    python -m src.run_all                    # linear SCM sweep
    python -m src.run_all --nonlinear        # full nonlinear sweep
    python -m src.run_all --interaction      # joint pathology sweep
    python -m src.run_all --all              # all experiments
    python -m src.run_all --quick            # fewer seeds for quick iteration
    python -m src.run_all --figures-only     # skip experiments, re-plot from CSVs
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from src.experiments import (
    run_all_sweeps,
    run_interaction_sweep,
)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use 3 seeds instead of 10 (faster iteration).")
    parser.add_argument("--figures-only", action="store_true",
                        help="Skip experiments; re-plot from existing CSVs.")
    parser.add_argument("--n-seeds", type=int, default=None,
                        help="Override number of seeds.")
    parser.add_argument("--nonlinear", action="store_true",
                        help="Run the nonlinear SCM sweep.")
    parser.add_argument("--interaction", action="store_true",
                        help="Run the joint dropout x confounders x density sweep.")
    parser.add_argument("--interaction-seeds", type=int, default=5,
                        help="Number of seeds for the interaction sweep.")
    parser.add_argument("--all", dest="run_all", action="store_true",
                        help="Run linear + nonlinear + interaction sweeps.")
    args = parser.parse_args()
    # With no experiment flags, run the main linear sweep.
    any_specific = args.nonlinear or args.interaction
    run_linear = args.run_all or not any_specific
    run_nonlinear = args.run_all or args.nonlinear
    run_interaction = args.run_all or args.interaction

    n_seeds = args.n_seeds if args.n_seeds is not None else (3 if args.quick else 10)
    interaction_seeds = (3 if args.quick else args.interaction_seeds)

    results_lin = Path("results/results.csv")
    results_nonlin = Path("results/results_nonlinear.csv")
    results_interaction = Path("results/results_interaction.csv")
    # --- Linear sweep ---
    if args.figures_only or not run_linear:
        assert results_lin.exists(), f"No cached results at {results_lin}"
        df_lin = pd.read_csv(results_lin)
    else:
        df_lin = run_all_sweeps(n_seeds=n_seeds, out_path=str(results_lin),
                                scm_type="linear")
    # --- Nonlinear sweep ---
    df_nonlin = None
    if run_nonlinear:
        if args.figures_only:
            assert results_nonlin.exists(), f"No cached nonlinear results at {results_nonlin}"
            df_nonlin = pd.read_csv(results_nonlin)
        else:
            df_nonlin = run_all_sweeps(n_seeds=n_seeds, out_path=str(results_nonlin),
                                       scm_type="nonlinear")
    # --- Interaction sweep (dropout x confounders x density) ---
    df_interaction = None
    if run_interaction:
        if args.figures_only:
            assert results_interaction.exists(), \
                f"No cached interaction results at {results_interaction}"
            df_interaction = pd.read_csv(results_interaction)
        else:
            df_interaction = run_interaction_sweep(
                n_seeds=interaction_seeds,
                out_path=str(results_interaction),
                scm_type="linear",
            )
    # --- Figures and tables ---
    from src.plotting import build_all

    build_all(df_lin, df_nonlin=df_nonlin, df_interaction=df_interaction)

if __name__ == "__main__":
    main()
