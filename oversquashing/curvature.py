"""Extended Forman-Ricci curvature (EFC) on the influence graph.

Generic: only consumes `A_col()` from an `InfluenceGraph` (same style as
`spectral.py` / `sensitivity.py`).

Simplification used here (verified, not assumed): for this repo's specific
relational construction, the off-diagonal part of Ã (`A_vertical +
A_horizontal`) is always symmetric -- boundary/coboundary are exact
transposes of each other, and each same-rank adjacency block is BᵀB/BBᵀ,
hence symmetric (checked in `scripts/influence_graph_sanity.py`). `A_col()` is
exactly this matrix (its `symmetrize()` step is a no-op here). So the
"directed" EFC(τ→σ) of Taha et al. collapses to the standard undirected
weighted Forman-Ricci curvature per edge {σ, τ} for every baseline in this
study: EFC(σ,τ) == EFC(τ,σ). This is a property of *this codebase's*
influence-graph construction, not a general property of the relational
framework -- worth re-checking before reusing this module for a lifting
whose relations are not built the same way (e.g. a genuinely asymmetric
rewiring relation).

w_T (triangles) is exact: it is (A^2)_{σ,τ} -- the zero diagonal of A
already forces every degenerate term out of the sum.

w_F (weighted "square"/4-cycle count) started life as the naive proxy
`(A^3)_{σ,τ}` (weighted closed walks of length 3 through the edge), but an
early run of `scripts/curvature_by_lifting.py` on MUTAG showed `frac_negative == 0.0`
for every single lifting -- an implausible signature for Forman curvature,
whose entire point is to expose negative-curvature bottlenecks. Root cause:
`(A^3)_{σ,τ}` counts *closed walks*, including degenerate ones that revisit
an endpoint (a=σ or b=τ in the walk τ→a→b→σ) rather than tracing a genuine
simple 4-cycle, which inflates w_F (and hence EFC) with spurious positive
mass. Fixed by inclusion-exclusion (`extended_forman_curvature`): subtract
`edge_w * (A²)_{σ,σ}` and `edge_w * (A²)_{τ,τ}` (walks revisiting σ or τ
respectively), add back `edge_w**3` once (the doubly-subtracted overlap).
What remains is an exact count of simple 4-cycles σ-a-b-τ-σ with a≠b and
a,b ∉ {σ,τ}. Still worth cross-checking against Taha et al.'s primary source
if available -- this derivation was worked out from first principles here,
not copied from the paper.

Second finding after the w_F fix (also empirical, also worth flagging):
`frac_negative` is *still* ~0 for every lifting on MUTAG. Root cause this
time is structural, not a bug: every original-graph edge (u,v) has a "free"
triangle through its own edge-cell in `A_col` (u and v are each vertical
neighbors of edge-cell e=(u,v), and horizontal neighbors of each other --
u-e-v is a genuine triangle), contributing +3 to that edge's EFC regardless
of anything else in the graph. Merging vertical and horizontal relations
into one `A_col` (the "one graph, one matrix" idea this whole study is built
on) is deliberate, not a mistake -- but it means the sign of EFC computed
this way is **not directly comparable to plain-graph Forman-Ricci
intuitions** (where F=0 is typical for a simple cycle and negative curvature
flags genuine bottlenecks). Documented as an open interpretive caveat here
rather than "fixed" by inventing a different formula without the primary
source to check against.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp


def weighted_degree(A: sp.csr_matrix) -> np.ndarray:
    """Weighted degree per cell (row sum of `A_col`; symmetric, so row sum
    == column sum)."""
    return np.asarray(A.sum(axis=1)).ravel()


def extended_forman_curvature(A: sp.csr_matrix) -> sp.csr_matrix:
    """EFC per edge of `A` (`A_col`: symmetric, weighted, zero diagonal).

    Returns a sparse matrix with the same sparsity pattern as `A`: entry
    (sigma, tau) is EFC({sigma, tau}) wherever `A[sigma, tau] > 0`.
    """
    w = weighted_degree(A)
    A2 = (A @ A).tocsr()  # w_T: weighted triangle count, exact (zero diagonal
    # of A already forces every (sigma,delta,tau) term with delta in {sigma,tau} to 0)
    A3 = (A2 @ A).tocsr()  # length-3 closed walks tau->a->b->sigma, raw (biased, see below)

    A_coo = A.tocoo()
    rows, cols = A_coo.row, A_coo.col
    edge_w = A_coo.data
    w_T = np.asarray(A2[rows, cols]).ravel()

    # Raw (A^3)_{sigma,tau} counts every closed walk tau->a->b->sigma, including
    # degenerate ones that revisit an endpoint (a=sigma or b=tau) rather than
    # tracing a simple 4-cycle -- this was checked empirically (an earlier,
    # uncorrected version of this function produced EFC with *zero* negative
    # entries on every MUTAG lifting, an implausible signature for Forman
    # curvature, whose whole point is to expose negative-curvature
    # bottlenecks). Removed here by inclusion-exclusion (A is symmetric, zero
    # diagonal): terms with a=sigma contribute edge_w * (A^2)_{sigma,sigma};
    # terms with b=tau contribute edge_w * (A^2)_{tau,tau}; both simultaneously
    # (a=sigma and b=tau) is edge_w^3 and gets double-subtracted, so it is
    # added back once. What remains after subtracting is an exact count of
    # simple 4-cycles sigma-a-b-tau-sigma (a != b, a,b not in {sigma,tau}).
    diag_A2 = A2.diagonal()
    raw_w_F = np.asarray(A3[rows, cols]).ravel()
    w_F = raw_w_F - edge_w * diag_A2[rows] - edge_w * diag_A2[cols] + edge_w**3

    efc_values = 4.0 - w[rows] - w[cols] + 3.0 * w_T + 2.0 * w_F
    return sp.csr_matrix((efc_values, (rows, cols)), shape=A.shape)


def curvature_summary(efc: sp.csr_matrix) -> dict:
    """Descriptive summary of the EFC distribution, one sample per undirected
    edge (upper triangle only, since EFC is symmetric here)."""
    coo = sp.triu(efc, k=1).tocoo()
    values = coo.data
    if values.size == 0:
        return {
            "n_edges": 0,
            "efc_mean": float("nan"),
            "efc_std": float("nan"),
            "frac_negative": float("nan"),
        }
    return {
        "n_edges": int(values.size),
        "efc_mean": float(values.mean()),
        "efc_std": float(values.std()),
        "frac_negative": float((values < 0).mean()),
    }


def betweenness_weighted_curvature(A: sp.csr_matrix, efc: sp.csr_matrix) -> dict:
    """wc = sum_e bc(e) * curv(e), nwc = sum_{e: curv(e)<0} bc(e) * curv(e)
    (Taha et al. Appendix D.2) -- a global curvature summary that stays
    robust to the number of cells a lifting adds (unlike raw mean curvature
    or λ2).

    Design decision: networkx's edge betweenness interprets `weight` as a
    *distance*, but our edge weights are influence *strength* -- so we pass
    `1/weight` as the distance (stronger influence = shorter path = more
    likely to lie on shortest paths). Not given explicitly by the source,
    documented here as the convention used.
    """
    G = nx.from_scipy_sparse_array(A)
    for _, _, data in G.edges(data=True):
        data["distance"] = 1.0 / data["weight"] if data["weight"] > 0 else np.inf
    bc = nx.edge_betweenness_centrality(G, weight="distance")

    wc, nwc = 0.0, 0.0
    for (u, v), b in bc.items():
        curv = efc[u, v] if efc[u, v] != 0 else efc[v, u]
        wc += b * curv
        if curv < 0:
            nwc += b * curv
    return {"wc": float(wc), "nwc": float(nwc)}


def paired_efc_on_shared_edges(
    efc_before: sp.csr_matrix, efc_after: sp.csr_matrix, edges: list
) -> list:
    """Before/after (paired) EFC comparison restricted to base-graph edges
    (pairs of node indices) present in both structures. Only meaningful for
    liftings that preserve the
    original 1-skeleton (cycle, clique, or the GNN_brut control) -- the
    caller is responsible for only passing such edges/liftings.
    """
    rows = []
    for u, v in edges:
        efc_b = efc_before[u, v] if efc_before[u, v] != 0 else efc_before[v, u]
        efc_a = efc_after[u, v] if efc_after[u, v] != 0 else efc_after[v, u]
        rows.append(
            {"u": int(u), "v": int(v), "efc_base": float(efc_b), "efc_lift": float(efc_a), "delta": float(efc_a - efc_b)}
        )
    return rows
