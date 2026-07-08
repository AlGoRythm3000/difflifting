"""Building the influence graph (the block everything else depends on).

Implements the relational framework of Taha et al. (ICLR 2025). For a
complex/lifted structure K,
described only by its chain of incidences [B_1, ..., B_D]
(B_k : (k-1)-cells -> k-cells, TopoNetX convention), we assemble:

- Ã^{R_i} for each relation R_i (id, boundary, coboundary, lower-adj, upper-adj);
- Ã = Σ_i Ã^{R_i}, the aggregated influence matrix (eq. 6);
- B = γI + Ã, the augmented matrix (eq. 7);
- A_col, the collapsed undirected adjacency (Def. 4.1).

This construction only depends on the incidences: it is therefore identical
whether they come from TopoNetX (cellular/simplicial liftings) or are
hand-assembled (hypergraph liftings) -- cf. `lift_adapter.py`.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def to_scipy(mat) -> sp.csr_matrix:
    """Coerce a dense/sparse torch tensor (or scipy/numpy array) to scipy CSR."""
    if hasattr(mat, "is_sparse") and mat.is_sparse:
        mat = mat.coalesce()
        idx = mat.indices().detach().cpu().numpy()
        val = mat.values().detach().cpu().numpy()
        return sp.csr_matrix((val, (idx[0], idx[1])), shape=tuple(mat.shape))
    if hasattr(mat, "detach"):  # dense torch tensor
        mat = mat.detach().cpu().numpy()
    return sp.csr_matrix(mat)


def _zero_diag(mat: sp.csr_matrix) -> sp.csr_matrix:
    mat = mat.tolil()
    mat.setdiag(0)
    return mat.tocsr()


def symmetrize(mat: sp.csr_matrix) -> sp.csr_matrix:
    """(Q+Qᵀ)/2, diagonal zeroed out (relational_oversquashing.md §4).

    Idempotent on an already-symmetric matrix (which is the case for
    `A_vertical`, `A_horizontal` and every `horizontal_block` by
    construction: boundary/coboundary are transposed blocks of each other,
    and each same-rank adjacency block is of the form BᵀB or BBᵀ, hence
    symmetric). We still apply the general formula here rather than assume
    symmetry, to stay correct if a future weighted/signed variant breaks
    this property.
    """
    sym = ((mat + mat.T) / 2.0).tocsr()
    return _zero_diag(sym)


def _block_matrix(blocks: dict, rank_sizes: list[int]) -> sp.csr_matrix:
    """Assemble a dict {(row_rank, col_rank): submatrix} into one big block matrix."""
    total = int(sum(rank_sizes))
    offsets = np.cumsum([0] + list(rank_sizes))
    M = sp.lil_matrix((total, total))
    for (i, j), block in blocks.items():
        if block is None or block.nnz == 0:
            continue
        r0, r1 = offsets[i], offsets[i + 1]
        c0, c1 = offsets[j], offsets[j + 1]
        M[r0:r1, c0:c1] = block
    return M.tocsr()


@dataclass
class InfluenceGraph:
    """Influence graph G(S, B) of a lifted structure K (Def. 3.1, Taha et al.).

    S = union of cells across all ranks. All matrices are |S| x |S|, indexed
    by blocks via `rank_slices` (one slice per rank, in rank order:
    0 = nodes, 1 = edges/hyperedges, 2 = faces, ...).
    """

    rank_sizes: list[int]
    rank_slices: list[slice]
    A_id: sp.csr_matrix
    A_boundary: sp.csr_matrix
    A_coboundary: sp.csr_matrix
    A_lower_adj: sp.csr_matrix
    A_upper_adj: sp.csr_matrix

    @property
    def n_cells(self) -> int:
        return int(sum(self.rank_sizes))

    @property
    def A_tilde(self) -> sp.csr_matrix:
        """Ã = Ã_id + Ã_boundary + Ã_coboundary + Ã_lower-adj + Ã_upper-adj (eq. 6)."""
        return (
            self.A_id + self.A_boundary + self.A_coboundary + self.A_lower_adj + self.A_upper_adj
        ).tocsr()

    @property
    def A_vertical(self) -> sp.csr_matrix:
        """Ã_boundary + Ã_coboundary (inter-rank: boundary/coboundary incidences)."""
        return (self.A_boundary + self.A_coboundary).tocsr()

    @property
    def A_horizontal(self) -> sp.csr_matrix:
        """Ã_lower-adj + Ã_upper-adj (intra-rank: lower/upper adjacencies)."""
        return (self.A_lower_adj + self.A_upper_adj).tocsr()

    def B(self) -> sp.csr_matrix:
        """Augmented matrix B = γI + Ã, γ = max row sum of Ã (eq. 7)."""
        A = self.A_tilde
        row_sums = np.asarray(A.sum(axis=1)).ravel()
        gamma = float(row_sums.max()) if row_sums.size else 0.0
        return (gamma * sp.identity(self.n_cells, format="csr") + A).tocsr()

    def A_col(self) -> sp.csr_matrix:
        """Collapsed adjacency: symmetrizes Ã (excluding identity) into an
        undirected weighted graph, across all ranks (Def. 4.1)."""
        return symmetrize(self.A_vertical + self.A_horizontal)

    def horizontal_block(self, rank: int) -> sp.csr_matrix:
        """Horizontal (k,k) sub-block for a given rank (node-node, edge-edge, ...)."""
        s = self.rank_slices[rank]
        return self.A_horizontal[s, s]

    def vertical_block(self, rank_hi: int, rank_lo: int) -> sp.csr_matrix:
        """Vertical (boundary+coboundary) sub-block between two adjacent ranks."""
        s_hi, s_lo = self.rank_slices[rank_hi], self.rank_slices[rank_lo]
        return self.A_vertical[np.ix_(range(s_hi.start, s_hi.stop), range(s_lo.start, s_lo.stop))]


def build_influence_graph(incidences: list) -> InfluenceGraph:
    """Builds the influence graph from the chain of incidences.

    Parameters
    ----------
    incidences : list
        incidences[k] = B_{k+1} (dense/sparse torch, scipy, or numpy), the
        boundary matrix from (k+1)-cells to k-cells, shape
        (n_k, n_{k+1}) -- TopoNetX convention. E.g. [B_1] for a hypergraph
        or a raw graph (ranks 0/1); [B_1, B_2] for a rank 0/1/2 complex.

    Returns
    -------
    InfluenceGraph
    """
    if len(incidences) == 0:
        raise ValueError("At least one incidence matrix B_1 is required (ranks 0 and 1).")

    incidences = [to_scipy(B) for B in incidences]
    max_rank = len(incidences)
    rank_sizes = [incidences[0].shape[0]] + [B.shape[1] for B in incidences]

    boundary, coboundary, lower_adj, upper_adj = {}, {}, {}, {}
    for k in range(1, max_rank + 1):
        B_k = incidences[k - 1]
        # R2 = boundary: message from the (k-1)-cell (boundary) to the k-cell.
        boundary[(k, k - 1)] = B_k.T.tocsr()
        # R3 = coboundary: message from the k-cell (coboundary) to the (k-1)-cell.
        coboundary[(k - 1, k)] = B_k.tocsr()
        # R4 = lower adjacency at rank k: k-cells sharing a (k-1)-face.
        lower_adj[(k, k)] = _zero_diag((B_k.T @ B_k).tocsr())
        # R5 = upper adjacency at rank k-1: (k-1)-cells sharing a k-coface.
        upper_adj[(k - 1, k - 1)] = _zero_diag((B_k @ B_k.T).tocsr())

    A_id = sp.identity(int(sum(rank_sizes)), format="csr")
    A_boundary = _block_matrix(boundary, rank_sizes)
    A_coboundary = _block_matrix(coboundary, rank_sizes)
    A_lower_adj = _block_matrix(lower_adj, rank_sizes)
    A_upper_adj = _block_matrix(upper_adj, rank_sizes)

    offsets = np.cumsum([0] + list(rank_sizes))
    rank_slices = [slice(int(offsets[i]), int(offsets[i + 1])) for i in range(len(rank_sizes))]

    return InfluenceGraph(
        rank_sizes=list(rank_sizes),
        rank_slices=rank_slices,
        A_id=A_id,
        A_boundary=A_boundary,
        A_coboundary=A_coboundary,
        A_lower_adj=A_lower_adj,
        A_upper_adj=A_upper_adj,
    )
