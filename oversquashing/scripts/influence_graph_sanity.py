"""Influence graph sanity check on MUTAG.

Checks, on a few MUTAG graphs, that `build_influence_graph` produces
matrices consistent with the theoretical framework:

- per-rank block sizes consistent with the input incidences;
- Ã_boundary and Ã_coboundary are transposes of each other (same relation,
  opposite direction);
- Ã_lower-adj and Ã_upper-adj are symmetric and have a zero diagonal;
- Ã has entries >= 0;
- gamma = max row sum of Ã, and B = gamma*I + Ã has a strictly positive
  diagonal (otherwise the sensitivity bound would be vacuous);
- A_col is symmetric, with a zero diagonal.

Nothing is trained here: this is a purely structural building block.
Usage: `python -m oversquashing.scripts.influence_graph_sanity` from the
repo root, in the `difflifting` conda env.
"""

import os.path as osp

import numpy as np
from torch_geometric.datasets import TUDataset

from oversquashing.lift_adapter import build_influence_graph_from_lifting
from tools.lifting.cycle_lifting import CellCycleLifting
from tools.lifting.hypergraph import HypergraphKHopLifting

DATA_ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "DATA", "DATASETS", "MUTAG")
N_GRAPHS = 5


def check_influence_graph(name: str, ig, incidences) -> None:
    print(f"\n--- {name} ---")
    print("rank_sizes:", ig.rank_sizes, " (n_cells total =", ig.n_cells, ")")

    expected_rank_sizes = [incidences[0].shape[0]] + [B.shape[1] for B in incidences]
    assert ig.rank_sizes == expected_rank_sizes, (
        f"rank_sizes {ig.rank_sizes} != expected from incidences {expected_rank_sizes}"
    )

    # Ã_boundary and Ã_coboundary must be transposes of each other (the same
    # boundary/coboundary relation, read in both directions).
    diff = (ig.A_boundary.T - ig.A_coboundary)
    assert diff.nnz == 0 or np.allclose(diff.toarray(), 0), "Ã_boundary^T != Ã_coboundary"

    # Upper/lower adjacencies: symmetric, zero diagonal.
    for label, mat in [("lower_adj", ig.A_lower_adj), ("upper_adj", ig.A_upper_adj)]:
        assert (mat - mat.T).nnz == 0 or np.allclose((mat - mat.T).toarray(), 0), f"Ã_{label} not symmetric"
        assert mat.diagonal().sum() == 0, f"Ã_{label} has a nonzero diagonal"

    A_tilde = ig.A_tilde
    assert A_tilde.min() >= 0, "Ã must have entries >= 0"

    row_sums = np.asarray(A_tilde.sum(axis=1)).ravel()
    gamma = float(row_sums.max())
    B = ig.B()
    # Ã_id contributes exactly 1 across the whole diagonal of Ã (no other
    # relation touches the diagonal) => diag(B) = gamma + 1 everywhere.
    diag_B = B.diagonal()
    assert np.allclose(diag_B, gamma + 1.0), "B's diagonal is inconsistent with gamma + diag(Ã_id)"
    print(f"gamma (max row sum of Ã) = {gamma:.3f}")
    print(f"Ã shape = {A_tilde.shape}, nnz = {A_tilde.nnz}")
    print(f"B shape = {B.shape}, nnz = {B.nnz}")

    A_col = ig.A_col()
    assert (A_col - A_col.T).nnz == 0 or np.allclose((A_col - A_col.T).toarray(), 0), "A_col not symmetric"
    assert A_col.diagonal().sum() == 0, "A_col has a nonzero diagonal"
    print(f"A_col shape = {A_col.shape}, nnz = {A_col.nnz} (symmetric, no diagonal: OK)")

    vert_nnz = ig.A_vertical.nnz
    horiz_nnz = ig.A_horizontal.nnz
    print(f"VERTICAL/HORIZONTAL decomposition nnz = {vert_nnz}, {horiz_nnz}")


def main() -> None:
    dataset = TUDataset(root=DATA_ROOT, name="MUTAG")
    print(f"MUTAG loaded: {len(dataset)} graphs")

    cell_lifting = CellCycleLifting()
    hyper_lifting = HypergraphKHopLifting(k_value=1)

    for i in range(min(N_GRAPHS, len(dataset))):
        data = dataset[i]
        print(f"\n=== Graph {i}: {data.num_nodes} nodes, {data.num_edges} edges (directed) ===")

        from oversquashing.lift_adapter import get_incidences_from_lifted_data

        lifted_cell = cell_lifting.lift_topology(data.clone())
        incidences_cell = get_incidences_from_lifted_data(lifted_cell)
        ig_cell = build_influence_graph_from_lifting(cell_lifting, data.clone())
        check_influence_graph("CellCycleLifting (cellular, ranks 0/1/2)", ig_cell, incidences_cell)

        lifted_hyper = hyper_lifting.lift_topology(data.clone())
        incidences_hyper = get_incidences_from_lifted_data(lifted_hyper)
        ig_hyper = build_influence_graph_from_lifting(hyper_lifting, data.clone())
        check_influence_graph("HypergraphKHopLifting (hypergraph, ranks 0/1)", ig_hyper, incidences_hyper)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
