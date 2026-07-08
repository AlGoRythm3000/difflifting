"""Structural sensitivity via the powers of the augmented matrix B.

Implements the sensitivity bound of Taha et al. (Lemma 3.2). Generic: only
consumes sparse matrices from an `InfluenceGraph` (same spirit as
`spectral.py`), so it's reusable without modification for any lifting/
domain.

Design decision: the `(B^t)` powers are computed on a **row-stochastic**
version of B (`normalized_B`),
not on Lemma 3.2's raw B. Raw B has a diagonal `gamma + 1` where `gamma`
depends on the graph (number/weight of the relations): its powers explode/
decay at scales that aren't comparable from one graph or lifting to
another. The normalized version keeps every entry in [0, 1] and makes
`(B^t)_{sigma,tau}` directly comparable across structures -- this is what's
used for every metric derived (d_inf, ΔS) in this module. Raw B
(`InfluenceGraph.B()`) stays available for anyone who wants Lemma 3.2's
literal bound.

Interpretation trap worth knowing (not a bug -- an expected effect of
row-stochastic normalization, not to be confused with a genuine loss of
signal): because B_hat is row-normalized, **adding higher-order cells
dilutes the transition mass** of cells that were already close (more
neighbors => less weight per neighbor), so ΔS_highorder =
S_complex - S_rewired can be **negative** for node-node pairs that are
already close in the base graph, even if the same cell addition creates
genuine shortcuts elsewhere for distant pairs (ΔS_highorder close to 0 or
positive at large geodesic distance). Confirmed empirically on
MUTAG/`CellCycleLifting` (initial run): ΔS_highorder(t=3) ~ -0.033 at
base distance 1, ~0 by distance 4. Always read ΔS_highorder as a function
of base geodesic distance, never as a single global scalar aggregated over
every pair.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

from oversquashing.influence_graph import InfluenceGraph


def normalized_B(ig: InfluenceGraph) -> sp.csr_matrix:
    """B_hat = D^{-1} B, row-stochastic (D = diag of B's row sums).

    A zero-sum row can't happen here: B = gamma*I + Ã has a diagonal
    gamma+1 >= 1 everywhere (cf. `influence_graph_sanity.py`), so every row
    has a sum >= 1.
    """
    B = ig.B()
    row_sums = np.asarray(B.sum(axis=1)).ravel()
    inv = np.where(row_sums > 0, 1.0 / row_sums, 0.0)
    return (sp.diags(inv) @ B).tocsr()


def bt_matrix(B_hat: sp.csr_matrix, t: int) -> np.ndarray:
    """(B_hat)^t, dense (the influence graphs here are small: at most a few
    hundred cells, so a dense `matrix_power` is easily fast enough and
    avoids re-implementing an iterative sparse power)."""
    if t < 1:
        raise ValueError("t must be >= 1")
    dense = B_hat.toarray()
    return np.linalg.matrix_power(dense, t)


def bt_matrix_sequence(B_hat: sp.csr_matrix, t_max: int) -> list[np.ndarray]:
    """[(B_hat)^1, ..., (B_hat)^t_max], via successive multiplications (a
    single pass, not t_max independent calls to `matrix_power`) -- to sweep
    t across a range of depths."""
    if t_max < 1:
        raise ValueError("t_max must be >= 1")
    dense = B_hat.toarray()
    powers = [dense]
    for _ in range(t_max - 1):
        powers.append(powers[-1] @ dense)
    return powers


def influence_distance(ig: InfluenceGraph) -> np.ndarray:
    """d_inf(sigma, tau) = smallest t such that (B^t)_{sigma,tau} > 0.

    Equivalent to the **directed, unweighted** shortest-path distance on the
    off-diagonal support of Ã (Ã_id only contributes to the diagonal, so it
    doesn't affect distances between distinct cells). Computed via shortest
    path rather than by sweeping t with matrix powers (much cheaper, and
    exact -- no t_max truncation).

    Returns an `n_cells x n_cells` float matrix, `np.inf` if tau is never
    reachable from sigma.
    """
    support = (ig.A_vertical + ig.A_horizontal).tocsr()
    support = support.astype(bool).astype(np.float64)
    return shortest_path(support, method="D", directed=True, unweighted=True)


def base_geodesic_distance(ig_base: InfluenceGraph) -> np.ndarray:
    """Undirected shortest-path distances between nodes of the base graph
    (ablation level (a), rank 0 only) -- used to order/bucket node-node
    pairs in the deltas (§4.2).

    `ig_base` must be an influence graph with ranks 0/1 only (base, no rank
    >= 2 cell): we restrict directly to the node-node block of A_col
    (undirected adjacency), no need to sort by rank.
    """
    n_nodes = ig_base.rank_sizes[0]
    node_slice = ig_base.rank_slices[0]
    A_nodes = ig_base.A_col()[node_slice, node_slice]
    return shortest_path(A_nodes, method="D", directed=False, unweighted=True)


def node_block(matrix: np.ndarray, ig: InfluenceGraph) -> np.ndarray:
    """Restricts an `n_cells x n_cells` matrix (e.g. `bt_matrix`'s output)
    to the node-node block (rank 0), the only block comparable across
    `base`/`rewired`/`complex`."""
    s = ig.rank_slices[0]
    return matrix[s, s]
