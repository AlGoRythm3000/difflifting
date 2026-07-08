"""Effective resistance between fixed base-node pairs.

The primary size-robust comparison metric across liftings (same node pairs,
different lifted topology in between -- an apples-to-apples comparison
regardless of how many cells a lifting adds).

Caveat worth flagging: this is effective resistance on the *undirected
symmetrized* `A_col`, which discards the influence graph's direction. A
genuinely directed effective resistance would need the pseudo-inverse of a
directed Laplacian -- out of scope, noted here as a possible extension, not
implemented.
"""

import numpy as np
import scipy.sparse as sp


def effective_resistance_matrix(A_col: sp.csr_matrix) -> np.ndarray:
    """R_eff(i, j) for every pair of cells, via the Moore-Penrose
    pseudo-inverse of the combinatorial Laplacian `L = D - A_col` (Klein &
    Randic's formula: `R_eff(i,j) = L^+_ii + L^+_jj - 2*L^+_ij`). Dense: the
    influence graphs studied here are small enough for this to be cheap.
    """
    A = np.asarray(A_col.todense(), dtype=float) if sp.issparse(A_col) else np.asarray(A_col, dtype=float)
    degrees = A.sum(axis=1)
    L = np.diag(degrees) - A
    L_pinv = np.linalg.pinv(L)
    diag = np.diag(L_pinv)
    return diag[:, None] + diag[None, :] - 2 * L_pinv
