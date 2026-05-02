"""
Synthetic single-cell GRN simulator with controllable pathologies.

We simulate from a linear additive-noise SCM (an approximation of mass-action
kinetics at steady state). This is the same substrate used by NOTEARS and
most causal-discovery papers. We then add scRNA-seq-specific pathologies on
top:
  - dropout        : per-cell, per-gene zero-inflation (technical noise)
  - n_confounders  : unobserved latent variables that drive groups of genes
  - mix_ratio      : samples drawn from a mixture of two SCMs (cell types)
  - feedback       : probability of adding a back-edge, producing cycles
  - density        : edge probability of the ground-truth DAG
  - pseudotime     : non-stationary parameter drift along a trajectory

Each pathology is controlled by one scalar parameter so the sweeps are easy
to interpret.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass, field
from typing import Optional


# Ground-truth graph construction
def random_dag(n_genes: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """
    Scale-free-ish DAG on `n_genes` nodes with given edge density.

    Returns a weighted adjacency matrix W where W[i, j] != 0 means i -> j.
    Signs and magnitudes are sampled uniformly from [-1, -0.5] U [0.5, 1].
    Topological order is the natural node order (0 -> 1 -> ...).
    """
    W = np.zeros((n_genes, n_genes))
    # For scale-freeness, earlier nodes have higher in/out degree prior.
    for j in range(1, n_genes):
        # probability mass favoring a few hub parents
        parent_prior = 1.0 / (np.arange(j) + 1)
        parent_prior = parent_prior / parent_prior.sum()
        n_parents = rng.binomial(j, density)
        if n_parents == 0:
            continue
        parents = rng.choice(j, size=n_parents, replace=False, p=parent_prior)
        for p in parents:
            sign = rng.choice([-1, 1])
            mag = rng.uniform(0.5, 1.0)
            W[p, j] = sign * mag
    return W

def add_feedback(W: np.ndarray, feedback: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add back-edges with probability `feedback` for each forward edge.

    Produces a (possibly) cyclic directed graph. We solve the fixed point
    with a Neumann series, so effective weights must have spectral radius < 1;
    back-edge magnitudes are kept small.
    """
    if feedback <= 0.0:
        return W
    W = W.copy()
    n = W.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] != 0 and rng.random() < feedback:
                sign = rng.choice([-1, 1])
                mag = rng.uniform(0.1, 0.3)  # keep small for stability
                W[j, i] = sign * mag
    # Safety: scale down if spectral radius >= 0.9
    rho = max(abs(np.linalg.eigvals(W)))
    if rho >= 0.9:
        W = W * (0.85 / rho)
    return W


# SCM sampler
def sample_linear_scm(
    W: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
    confounder_loadings: Optional[np.ndarray] = None,
    latent_samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample n cells from X = (I - W)^{-1} (E + L @ Z).

    W : weighted adjacency (i -> j means W[i, j])
    confounder_loadings : (n_genes, n_confounders) matrix L
    latent_samples       : (n_samples, n_confounders) matrix Z
    """
    n_genes = W.shape[0]
    E = rng.normal(0, noise_scale, size=(n_samples, n_genes))
    if confounder_loadings is not None and latent_samples is not None:
        E = E + latent_samples @ confounder_loadings.T
    # X (I - W) = E  =>  X = E (I - W)^{-1}
    # With cycles, (I - W) may be close to singular; we already constrained rho.
    X = np.linalg.solve((np.eye(n_genes) - W).T, E.T).T
    return X


def sample_nonlinear_scm(
    W: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
    confounder_loadings: Optional[np.ndarray] = None,
    latent_samples: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample from a nonlinear additive-noise SCM: X_j = tanh(sum_pa W[pa,j]*X_pa) + E_j.

    For DAGs: exact ancestral sampling via topological order.
    For cyclic graphs: fixed-point iteration (converges because spectral radius of W < 0.9
    and tanh is 1-Lipschitz, so the map is contractive when ||W||_2 < 1).
    Falls back to linear SCM if iteration does not converge within tolerance.
    """
    n_genes = W.shape[0]
    E = rng.normal(0, noise_scale, size=(n_samples, n_genes))
    if confounder_loadings is not None and latent_samples is not None:
        E = E + latent_samples @ confounder_loadings.T

    G = nx.DiGraph((np.abs(W) > 1e-10).astype(int))
    try:
        order = list(nx.topological_sort(G))
        # Ancestral sampling: exact for DAGs
        X = np.zeros((n_samples, n_genes))
        for j in order:
            parents = np.where(np.abs(W[:, j]) > 1e-10)[0]
            if len(parents) == 0:
                X[:, j] = E[:, j]
            else:
                z = X[:, parents] @ W[parents, j]
                X[:, j] = np.tanh(z) + E[:, j]
        return X
    except nx.NetworkXUnfeasible:
        # Cyclic graph: fixed-point iteration
        # Scale W so operator norm < 1 (guarantees contraction with tanh)
        op_norm = np.linalg.norm(W, ord=2)
        W_c = W * (0.85 / max(op_norm, 1e-10)) if op_norm >= 0.9 else W
        X = E.copy()
        for _ in range(300):
            X_new = np.tanh(X @ W_c) + E
            if np.max(np.abs(X_new - X)) < 1e-4:
                return X_new
            X = X_new
        # Convergence failed: fall back to linear
        return sample_linear_scm(W, n_samples, rng, noise_scale, confounder_loadings, latent_samples)


# scRNA-seq technical noise
def apply_dropout(X: np.ndarray, dropout: float, rng: np.random.Generator) -> np.ndarray:
    """
    Expression-dependent zero-inflation (MNAR model).

    P(zero | X_ij) = exp(-λ · (X_ij - X_min)), where λ is calibrated via
    binary search so that the marginal dropout rate equals `dropout`.
    Lowly-expressed entries drop out at higher rates, matching the empirical
    zero-inflation bias observed in scRNA-seq data.
    """
    if dropout <= 0.0:
        return X
    X_pos = X - X.min()  # shift so minimum expression maps to P(zero)=1
    # Binary search for λ: f(λ) = mean(exp(-λ·X_pos)) is decreasing in λ.
    lo, hi = 0.0, 500.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if np.mean(np.exp(-mid * X_pos)) > dropout:
            lo = mid
        else:
            hi = mid
    lam = (lo + hi) / 2.0
    prob = np.exp(-lam * X_pos)
    mask = rng.random(X.shape) < prob
    Xd = X.copy()
    Xd[mask] = 0.0
    return Xd

# Data classes for config and output
@dataclass
class SimConfig:
    n_genes: int = 50
    n_cells: int = 2000
    density: float = 0.08
    dropout: float = 0.0
    n_confounders: int = 0
    mix_ratio: float = 0.0          # 0 = single cell type, 0.5 = balanced mix of two
    feedback: float = 0.0
    pseudotime_drift: float = 0.0   # 0 = stationary
    noise_scale: float = 1.0
    seed: int = 0
    scm_type: str = "linear"        # "linear" or "nonlinear"

@dataclass
class SimOutput:
    X: np.ndarray           # (n_cells, n_genes) observed expression
    W_true: np.ndarray      # ground-truth signed adjacency
    A_true: np.ndarray      # ground-truth binary adjacency (i -> j)
    meta: dict = field(default_factory=dict)

def simulate(cfg: SimConfig) -> SimOutput:
    rng = np.random.default_rng(cfg.seed)
    sampler = sample_nonlinear_scm if cfg.scm_type == "nonlinear" else sample_linear_scm
    # 1) Build ground truth DAG (possibly with feedback).
    W = random_dag(cfg.n_genes, cfg.density, rng)
    W_true_for_eval = W.copy()  # we always evaluate against the acyclic skeleton
    W = add_feedback(W, cfg.feedback, rng)
    # 2) Latent confounders (observed cells are driven by unseen variables).
    if cfg.n_confounders > 0:
        L = rng.normal(0, 1.0, size=(cfg.n_genes, cfg.n_confounders))
        # each confounder influences ~30% of genes; zero out the rest
        mask = rng.random(L.shape) < 0.3
        L = L * mask
    else:
        L = None
    # 3) Sampling, possibly from a mixture of two SCMs (cell-type mixing).
    if cfg.mix_ratio > 0.0:
        n_a = int(cfg.n_cells * (1 - cfg.mix_ratio))
        n_b = cfg.n_cells - n_a
        W_b = random_dag(cfg.n_genes, cfg.density, rng)  # different cell-type graph
        Z_a = rng.normal(0, 1.0, size=(n_a, cfg.n_confounders)) if L is not None else None
        Z_b = rng.normal(0, 1.0, size=(n_b, cfg.n_confounders)) if L is not None else None
        Xa = sampler(W, n_a, rng, cfg.noise_scale, L, Z_a)
        Xb = sampler(W_b, n_b, rng, cfg.noise_scale, L, Z_b)
        X = np.concatenate([Xa, Xb], axis=0)
        rng.shuffle(X)  # shuffle so methods can't use ordering
    elif cfg.pseudotime_drift > 0.0:
        # non-stationary: weights drift linearly along pseudotime
        X_chunks = []
        n_chunks = 10
        per_chunk = cfg.n_cells // n_chunks
        for k in range(n_chunks):
            t = k / max(n_chunks - 1, 1)
            drift = 1.0 + cfg.pseudotime_drift * (t - 0.5)
            Z = rng.normal(0, 1.0, size=(per_chunk, cfg.n_confounders)) if L is not None else None
            X_chunks.append(sampler(W * drift, per_chunk, rng, cfg.noise_scale, L, Z))
        X = np.concatenate(X_chunks, axis=0)
    else:
        Z = rng.normal(0, 1.0, size=(cfg.n_cells, cfg.n_confounders)) if L is not None else None
        X = sampler(W, cfg.n_cells, rng, cfg.noise_scale, L, Z)
    # 4) Technical dropout.
    X = apply_dropout(X, cfg.dropout, rng)
    A_true = (np.abs(W_true_for_eval) > 1e-8).astype(int)
    return SimOutput(X=X, W_true=W_true_for_eval, A_true=A_true, meta={"cfg": cfg.__dict__})
