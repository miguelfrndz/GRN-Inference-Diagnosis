"""
Inference methods:
Each method takes expression matrix X of shape (n_cells, n_genes) and returns
a score matrix S of shape (n_genes, n_genes) where S[i, j] is the confidence
that i -> j. Undirected methods return symmetric scores.

Methods included:
    - pearson          : absolute Pearson correlation (undirected)
    - mutual_info      : mutual information via discretization (undirected)
    - genie3_lite      : random-forest feature importance (asymmetric)
    - pc_lite          : PC-algorithm skeleton via conditional independence
    - ges_lite         : greedy equivalence search on linear Gaussian score
    - notears          : NOTEARS linear with L1 (acyclicity as constraint)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations

# Correlation-based (undirected)
def pearson(X: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation. Symmetric, diagonal zeroed."""
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    S = np.abs(C)
    np.fill_diagonal(S, 0.0)
    return S

def _discretize(x: np.ndarray, n_bins: int = 6) -> np.ndarray:
    """Equal-frequency discretization."""
    ranks = np.argsort(np.argsort(x))
    return (ranks * n_bins // len(x)).astype(int)

def mutual_info(X: np.ndarray) -> np.ndarray:
    """Mutual information between discretized gene pairs (undirected)."""
    n_cells, n_genes = X.shape
    Xd = np.stack([_discretize(X[:, j]) for j in range(n_genes)], axis=1)
    S = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            # MI estimator on equal-frequency bins.
            xi = Xd[:, i]
            xj = Xd[:, j]
            bins = max(xi.max(), xj.max()) + 1
            joint = np.zeros((bins, bins))
            for a, b in zip(xi, xj):
                joint[a, b] += 1
            joint /= joint.sum()
            px = joint.sum(axis=1, keepdims=True)
            py = joint.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                mi_mat = joint * (np.log(joint + 1e-12) - np.log(px + 1e-12) - np.log(py + 1e-12))
            mi = np.nansum(mi_mat)
            S[i, j] = S[j, i] = max(mi, 0.0)
    return S


# GENIE3-lite (asymmetric, tree-based)
def genie3_lite(X: np.ndarray, n_estimators: int = 50, max_features: str = "sqrt") -> np.ndarray:
    """
    For each target gene j, regress X[:, j] on the other genes with a
    random forest; importance of feature i becomes S[i, j].
    """
    n_cells, n_genes = X.shape
    S = np.zeros((n_genes, n_genes))
    for j in range(n_genes):
        idx = [k for k in range(n_genes) if k != j]
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=0,
            n_jobs=1,
        )
        rf.fit(X[:, idx], X[:, j])
        imp = rf.feature_importances_
        for k_local, i in enumerate(idx):
            S[i, j] = imp[k_local]
    return S


# PC-lite (constraint-based)
def _partial_corr(X: np.ndarray, i: int, j: int, cond: list) -> float:
    """Partial correlation of (i, j) given `cond`, via regression residuals."""
    if not cond:
        return np.corrcoef(X[:, i], X[:, j])[0, 1]
    Z = X[:, cond]
    # regress out Z from both
    beta_i, *_ = np.linalg.lstsq(Z, X[:, i], rcond=None)
    beta_j, *_ = np.linalg.lstsq(Z, X[:, j], rcond=None)
    ri = X[:, i] - Z @ beta_i
    rj = X[:, j] - Z @ beta_j
    return np.corrcoef(ri, rj)[0, 1]

def _fisher_z_test(r: float, n: int, k: int, alpha: float = 0.05) -> bool:
    """Return True if correlation is significantly different from zero."""
    r = np.clip(r, -0.9999, 0.9999)
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(max(n - k - 3, 1))
    # two-tailed
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z) / se))
    return p < alpha

def pc_lite(X: np.ndarray, alpha: float = 0.05, max_cond: int = 2) -> np.ndarray:
    """
    Removes edges via conditional independence tests up to conditioning 
    set size `max_cond`, then returns the skeleton as symmetric 0/1 scores.
    """
    n_cells, n_genes = X.shape
    # start with complete undirected graph
    adj = np.ones((n_genes, n_genes)) - np.eye(n_genes)

    for k in range(0, max_cond + 1):
        # iterate over existing edges
        for i in range(n_genes):
            for j in range(i + 1, n_genes):
                if adj[i, j] == 0:
                    continue
                # candidate conditioning neighbors
                neighbors = [m for m in range(n_genes) if m != i and m != j and adj[i, m] == 1]
                if len(neighbors) < k:
                    continue
                found_sep = False
                for cond in combinations(neighbors, k):
                    r = _partial_corr(X, i, j, list(cond))
                    if not _fisher_z_test(r, n_cells, k, alpha):
                        # conditionally independent given `cond`
                        adj[i, j] = adj[j, i] = 0
                        found_sep = True
                        break
                if found_sep:
                    continue
    # Use |partial correlation given empty set| as a score on retained edges.
    S = np.zeros((n_genes, n_genes))
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    S = adj * np.abs(C)
    return S

# GES-lite (greedy forward pass, BIC score, acyclicity enforced)
def _bic_local(X: np.ndarray, j: int, parents: list) -> float:
    """BIC for regressing X[:, j] on X[:, parents]. Higher is better."""
    n, _ = X.shape
    y = X[:, j]
    if not parents:
        resid = y - y.mean()
        ssr = np.sum(resid ** 2)
        k = 1
    else:
        Z = X[:, parents]
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ssr = np.sum(resid ** 2)
        k = len(parents) + 1
    # Gaussian BIC (up to constants): -n/2 log(ssr/n) - k/2 log(n)
    return -0.5 * n * np.log(ssr / n + 1e-12) - 0.5 * k * np.log(n)

def ges_lite(X: np.ndarray, max_parents: int = 3) -> np.ndarray:
    """
    Minimalist GES greedy forward pass: for each node j, greedily
    add the parent that most improves local BIC, constrained to `max_parents`.
    Enforces acyclicity by only adding edges from ancestors with lower
    topological rank (we use variance-based initial ordering).
    """
    n_cells, n_genes = X.shape
    variances = X.var(axis=0)
    order = np.argsort(variances)  # proxy topological order
    S = np.zeros((n_genes, n_genes))
    rank = {node: r for r, node in enumerate(order)}
    for j in range(n_genes):
        candidates = [i for i in range(n_genes) if i != j and rank[i] < rank[j]]
        parents: list = []
        best_score = _bic_local(X, j, parents)
        improved = True
        while improved and len(parents) < max_parents:
            improved = False
            best_addition = None
            for c in candidates:
                if c in parents:
                    continue
                new_score = _bic_local(X, j, parents + [c])
                if new_score > best_score + 1e-6:
                    best_score = new_score
                    best_addition = c
                    improved = True
            if best_addition is not None:
                parents.append(best_addition)
        # score edges by their individual delta-BIC contribution
        for p in parents:
            others = [q for q in parents if q != p]
            delta = _bic_local(X, j, parents) - _bic_local(X, j, others)
            S[p, j] = max(delta, 0.0)
    return S


# NOTEARS linear with L1 regularization (optimization-based)
def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.05,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
) -> np.ndarray:
    """
    NOTEARS with linear model and L1 regularization.
    """
    n, d = X.shape
    X = X - X.mean(axis=0, keepdims=True)

    def _loss(W_flat):
        W = W_flat.reshape(d, d)
        R = X - X @ W
        loss = 0.5 / n * (R ** 2).sum()
        grad = -1.0 / n * X.T @ R
        return loss, grad

    def _h(W):
        # h(W) = trace(exp(W*W)) - d
        E = np.linalg.matrix_power(np.eye(d) + W * W / d, d)
        return (np.trace(E) - d), E.T * W * 2

    def _full(W_flat, rho, alpha):
        W = W_flat.reshape(d, d)
        loss, grad_loss = _loss(W_flat)
        h, grad_h = _h(W)
        obj = loss + 0.5 * rho * h ** 2 + alpha * h + lambda1 * np.abs(W).sum()
        grad = grad_loss + (rho * h + alpha) * grad_h
        return obj, W, h, grad

    from scipy.optimize import minimize

    W_est = np.zeros(d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        W_new, h_new = None, None
        while rho < rho_max:
            def _obj_grad(W_flat):
                obj, _W, _h, grad = _full(W_flat, rho, alpha)
                # L1 subgradient
                W = W_flat.reshape(d, d)
                grad = grad + lambda1 * np.sign(W)
                return obj, grad.flatten()
            sol = minimize(_obj_grad, W_est, jac=True, method="L-BFGS-B",
                           options={"maxiter": 50})
            W_new = sol.x
            h_new = _h(W_new.reshape(d, d))[0]
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        W_est, h = W_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = W_est.reshape(d, d)
    # Threshold very small values to zero for numerical stability.
    W_est[np.abs(W_est) < 0.1] = 0.0
    return np.abs(W_est)

METHODS = {
    "Pearson":   pearson,
    "MI":        mutual_info,
    "GENIE3":    genie3_lite,
    "PC":        pc_lite,
    "GES":       ges_lite,
    "NOTEARS":   notears_linear,
}

METHOD_CLASS = {
    "Pearson":   "correlation",
    "MI":        "correlation",
    "GENIE3":    "tree-ensemble",
    "PC":        "constraint-causal",
    "GES":       "score-causal",
    "NOTEARS":   "optimization-causal",
}
