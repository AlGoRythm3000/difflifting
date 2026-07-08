"""Spectral gap λ₂ and Cheeger constant on the influence graph.

Everything is computed on already-symmetric, non-negative matrices
(output of `InfluenceGraph.A_col()` / `symmetrize(...)`): that's the only
assumption this module needs, it knows nothing about the influence graph
itself (ranks, relations...), which makes it reusable by every other
structural comparison in this package.

Open item (decision made here):
- Normalized Laplacian L_sym by default (`normalized=True` everywhere).
- Symmetrization: A_col / (Q+Qᵀ)/2, not the unweighted version.

Interpretation trap worth knowing (not a bug -- a standard property of
L_sym): an isolated cell (degree 0) in a scope contributes an eigenvalue of
exactly **1** to L_sym (not 0), whatever convention is chosen for D^{-1/2}
on that row -- because the D^{-1/2} A D^{-1/2} term vanishes anyway
(A[i,:]=0), and only `I` contributes on the diagonal. If λ₂ ≈ 1 for a scope,
this is NOT necessarily a sign of good connectivity: it can come from an
isolated cell whose trivial eigenvalue (1) is smaller than the rest of the
graph's real spectral "gap" (often > 1 for a well-connected graph, e.g. K_n
has 2nd eigenvalue n/(n-1) > 1). Hence the `n_isolated` diagnostic in
`_scope_metrics`: always cross-check it against λ₂ ≈ 1 before concluding
good connectivity.
"""

import numpy as np
import scipy.sparse as sp

from oversquashing.influence_graph import InfluenceGraph, symmetrize


def normalized_laplacian(A: sp.csr_matrix) -> np.ndarray:
    """L_sym = I - D^{-1/2} A D^{-1/2}, dense (graphs here are small).

    Standard convention for zero-degree nodes (a cell isolated in this
    scope): set D^{-1/2}=0 on that row/column rather than divide by zero --
    the isolated cell then contributes an extra trivial 0 eigenvalue, which
    is the expected behavior (no information can transit through it in this
    scope).
    """
    A = np.asarray(A.todense(), dtype=float) if sp.issparse(A) else np.asarray(A, dtype=float)
    n = A.shape[0]
    degrees = A.sum(axis=1)
    with np.errstate(divide="ignore"):
        inv_sqrt_deg = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = np.diag(inv_sqrt_deg)
    return np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt


def combinatorial_laplacian(A: sp.csr_matrix) -> np.ndarray:
    """L = D - A (unnormalized alternative, cf. open decision §9)."""
    A = np.asarray(A.todense(), dtype=float) if sp.issparse(A) else np.asarray(A, dtype=float)
    D = np.diag(A.sum(axis=1))
    return D - A


def spectral_gap(A: sp.csr_matrix, normalized: bool = True) -> float:
    """λ₂ = 2nd smallest eigenvalue of the Laplacian.

    Returns NaN if the scope has fewer than 2 cells (λ₂ undefined).
    Note: λ₂ = 0 is a legitimate value (disconnected scope), not an error.
    """
    n = A.shape[0]
    if n < 2:
        return float("nan")
    L = normalized_laplacian(A) if normalized else combinatorial_laplacian(A)
    eigenvalues = np.linalg.eigvalsh(L)
    return float(eigenvalues[1])


def cheeger_bounds(lambda2: float) -> tuple[float, float]:
    """Cheeger bounds λ₂/2 ≤ h ≤ √(2λ₂) (valid for L_sym only)."""
    if np.isnan(lambda2):
        return float("nan"), float("nan")
    return lambda2 / 2.0, float(np.sqrt(max(2.0 * lambda2, 0.0)))


def fiedler_sweep_cheeger(A: sp.csr_matrix, normalized: bool = True) -> float:
    """Approximates the Cheeger constant via a Fiedler-vector sweep.

    Sorts cells by the value of the eigenvector associated with λ₂, then
    evaluates the conductance of every prefix cut; returns the observed
    minimum. Conductance(S) = cut(S, Sᶜ) / min(vol(S), vol(Sᶜ)).
    """
    n = A.shape[0]
    if n < 2:
        return float("nan")
    A_dense = np.asarray(A.todense(), dtype=float) if sp.issparse(A) else np.asarray(A, dtype=float)
    degrees = A_dense.sum(axis=1)
    total_vol = degrees.sum()
    if total_vol == 0:
        return 0.0

    L = normalized_laplacian(A_dense, ) if normalized else combinatorial_laplacian(A_dense)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler = eigenvectors[:, 1]

    order = np.argsort(fiedler)
    A_ordered = A_dense[np.ix_(order, order)]
    deg_ordered = degrees[order]

    vol_S = 0.0
    cut = 0.0
    best_h = float("inf")
    for i in range(n - 1):
        # Add cell i to S (the previous i are already in it, by construction
        # of the loop): the cut changes by -2*(weight i->S) + deg(i).
        weight_to_S = A_ordered[i, :i].sum()
        cut += deg_ordered[i] - 2.0 * weight_to_S
        vol_S += deg_ordered[i]
        vol_min = min(vol_S, total_vol - vol_S)
        if vol_min > 0:
            best_h = min(best_h, cut / vol_min)
    return float(best_h) if np.isfinite(best_h) else float("nan")


def _scope_metrics(A: sp.csr_matrix, normalized: bool = True) -> dict:
    lambda2 = spectral_gap(A, normalized=normalized)
    cheeger_lo, cheeger_hi = cheeger_bounds(lambda2) if normalized else (float("nan"), float("nan"))
    cheeger_sweep = fiedler_sweep_cheeger(A, normalized=normalized)
    degrees = np.asarray(A.sum(axis=1)).ravel()
    n_isolated = int((degrees == 0).sum())
    return {
        "n_cells": A.shape[0],
        "n_edges": int(A.nnz // 2),
        "n_isolated": n_isolated,
        "lambda2": lambda2,
        "cheeger_lo": cheeger_lo,
        "cheeger_hi": cheeger_hi,
        "cheeger_sweep": cheeger_sweep,
    }


RANK_NAMES = {0: "node", 1: "edge", 2: "face"}


def compute_scope_metrics(ig: InfluenceGraph, normalized: bool = True) -> dict[str, dict]:
    """Computes (n, λ₂, Cheeger bounds, Fiedler sweep) on the 4 scopes below:

    - "global": A_col(), the full collapsed influence graph;
    - "vertical": Ã_boundary + Ã_coboundary (already symmetric, cf. `symmetrize`);
    - "horizontal": Ã_lower-adj + Ã_upper-adj (same);
    - "horizontal_rank<k>": the horizontal (k,k) sub-block for each rank
      present in the structure (node-node, edge-edge, face-face...).

    Returns
    -------
    dict[str, dict]
        One entry per scope, each with n_cells/n_edges/lambda2/
        cheeger_lo/cheeger_hi/cheeger_sweep.
    """
    scopes = {
        "global": ig.A_col(),
        "vertical": symmetrize(ig.A_vertical),
        "horizontal": symmetrize(ig.A_horizontal),
    }
    for rank in range(len(ig.rank_sizes)):
        if ig.rank_sizes[rank] == 0:
            continue
        name = f"horizontal_rank{rank}" + (f"_{RANK_NAMES[rank]}" if rank in RANK_NAMES else "")
        scopes[name] = symmetrize(ig.horizontal_block(rank))

    return {name: _scope_metrics(A, normalized=normalized) for name, A in scopes.items()}
