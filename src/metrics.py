"""
Metrics: edge-recovery and error-type decomposition.

We use two primary metrics:
    - AUPRC (area under the precision-recall curve), computed on the
      *skeleton* (undirected ground truth) to be fair to symmetric methods.
    - AUPRC-directed, computed on the directed ground truth (only scored
      for methods that produce asymmetric output).

For error-type decomposition we binarize each method's output at a common
top-K threshold (K = number of true edges) and categorize each predicted
edge relative to the ground-truth DAG:
    - TRUE        : predicted edge matches a ground-truth directed edge
    - REVERSED    : predicted edge's reverse is a ground-truth edge
    - CONFOUNDED  : i, j share a common ancestor in ground truth but no edge
    - SPURIOUS    : neither of the above
And for missed edges:
    - MISSED      : ground-truth edge not recovered by the method
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_curve, auc

def _symmetrize(S: np.ndarray) -> np.ndarray:
    """Max over (i,j) and (j,i) to get undirected scores."""
    return np.maximum(S, S.T)

def auprc_undirected(S: np.ndarray, A_true: np.ndarray) -> float:
    """
    AUPRC on the undirected skeleton.
    """
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    skeleton = ((A_true + A_true.T) > 0).astype(int)[iu]
    scores = _symmetrize(S)[iu]
    if skeleton.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(skeleton, scores)
    return float(auc(r, p))

def auprc_directed(S: np.ndarray, A_true: np.ndarray) -> float:
    """
    AUPRC on the directed ground truth (off-diagonal only).
    For symmetric methods this will is inherently lower (random direction),
    which reflects their lack of orientation information.
    """
    n = S.shape[0]
    mask = ~np.eye(n, dtype=bool)
    y = A_true[mask].astype(int)
    s = S[mask]
    if y.sum() == 0:
        return float("nan")
    p, r, _ = precision_recall_curve(y, s)
    return float(auc(r, p))

def topk_edges(S: np.ndarray, k: int, directed: bool) -> np.ndarray:
    """Return binary adjacency matrix with top-k edges from S."""
    n = S.shape[0]
    if directed:
        scores = S.copy()
        np.fill_diagonal(scores, -np.inf)
        flat = scores.flatten()
        if k >= (flat > -np.inf).sum():
            thresh = -np.inf
        else:
            thresh = np.partition(flat, -k)[-k]
        A = (scores >= thresh).astype(int)
        np.fill_diagonal(A, 0)
    else:
        Ssym = _symmetrize(S)
        np.fill_diagonal(Ssym, -np.inf)
        iu = np.triu_indices(n, k=1)
        vals = Ssym[iu]
        if k >= len(vals):
            thresh = -np.inf
        else:
            thresh = np.partition(vals, -k)[-k]
        A = np.zeros_like(S, dtype=int)
        A[iu] = (vals >= thresh).astype(int)
        A = A + A.T
    return A

def _common_ancestors(A_true: np.ndarray, i: int, j: int) -> bool:
    """Do i and j share a common ancestor in the ground-truth DAG?"""
    G = nx.DiGraph(A_true)
    try:
        ai = nx.ancestors(G, i) | {i}
        aj = nx.ancestors(G, j) | {j}
    except Exception:
        return False
    return len(ai & aj) > 0 and (i not in aj) and (j not in ai)

def error_decomposition(
    S: np.ndarray,
    A_true: np.ndarray,
    directed: bool,
) -> dict:
    """
    Decompose predicted edges into error categories at top-K threshold.

    K = number of ground-truth (directed) edges. For undirected output we
    evaluate the skeleton and count FPs via the same logic symmetrized.
    """
    k_true = int(A_true.sum())
    if k_true == 0:
        return {"TRUE": 0, "REVERSED": 0, "CONFOUNDED": 0, "SPURIOUS": 0,
                "MISSED": 0, "K": 0}

    A_pred = topk_edges(S, k_true, directed=directed)
    n = A_true.shape[0]
    true_set = {(i, j) for i in range(n) for j in range(n) if A_true[i, j]}
    if directed:
        pred_set = {(i, j) for i in range(n) for j in range(n) if A_pred[i, j]}
    else:
        # treat undirected edges as the pair of directed edges (i<j) and (j<i)
        pred_set = {(i, j) for i in range(n) for j in range(n)
                    if i != j and A_pred[i, j]}
    counts = {"TRUE": 0, "REVERSED": 0, "CONFOUNDED": 0, "SPURIOUS": 0}
    G_true = nx.DiGraph(A_true)
    for (i, j) in pred_set:
        if (i, j) in true_set:
            counts["TRUE"] += 1
        elif (j, i) in true_set:
            counts["REVERSED"] += 1
        else:
            try:
                ai = nx.ancestors(G_true, i) | {i}
                aj = nx.ancestors(G_true, j) | {j}
                shares_ancestor = len(ai & aj) > 0
            except Exception:
                shares_ancestor = False
            if shares_ancestor:
                counts["CONFOUNDED"] += 1
            else:
                counts["SPURIOUS"] += 1
    if directed:
        missed = len(true_set - pred_set)
    else:
        undirected_pred = {frozenset([i, j]) for (i, j) in pred_set}
        undirected_true = {frozenset([i, j]) for (i, j) in true_set}
        missed = len(undirected_true - undirected_pred)
    counts["MISSED"] = missed
    counts["K"] = k_true
    return counts
