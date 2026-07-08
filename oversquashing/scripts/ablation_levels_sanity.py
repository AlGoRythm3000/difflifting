"""Sanity check for the 3-level ablation (base / rewired / complex).

Checks, on MUTAG, the expected property of `build_influence_graph_levels`:

- For `CellCycleLifting` (preserves the 1-skeleton): "rewired" == "base"
  restricted to the node-node block (ΔS_rewire = 0 by construction).
- For hypergraph liftings (ranks 0/1 only): "rewired" == "complex" exactly
  (truncating to `incidence_1` is a no-op, there's no `incidence_2` to drop).

Nothing is trained here: purely structural sanity check.
Usage: `python -m oversquashing.scripts.ablation_levels_sanity` from the
repo root, in the `difflifting` conda env.
"""

import os.path as osp

import numpy as np
from torch_geometric.datasets import TUDataset

from oversquashing.lift_adapter import build_influence_graph_levels
from tools.lifting.cycle_lifting import CellCycleLifting
from tools.lifting.hypergraph import HypergraphKHopLifting, HypergraphKNNLifting
from tools.lifting.kernel import HypergraphKernelLifting

DATA_ROOT = osp.join(
    osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "DATA", "DATASETS", "MUTAG"
)
N_GRAPHS = 5


def _assert_equal_node_block(mat_x, mat_y, label: str) -> None:
    diff = (mat_x - mat_y)
    max_abs_diff = np.abs(diff.toarray()).max() if diff.nnz else 0.0
    assert max_abs_diff < 1e-8, f"{label}: node-node block differs (max |diff| = {max_abs_diff})"


def check_cell_cycle_preserves_1_skeleton(data, graph_idx: int) -> None:
    lifting = CellCycleLifting()
    levels = build_influence_graph_levels(lifting, data.clone())
    ig_base, ig_rewired = levels["base"], levels["rewired"]

    assert ig_base.rank_sizes[0] == ig_rewired.rank_sizes[0], (
        f"graph {graph_idx} CellCycleLifting: node count differs between base and rewired"
    )
    assert ig_base.rank_sizes[1] == ig_rewired.rank_sizes[1], (
        f"graph {graph_idx} CellCycleLifting: edge count differs between base ({ig_base.rank_sizes[1]}) "
        f"and rewired ({ig_rewired.rank_sizes[1]}) -- the 1-skeleton shouldn't be modified"
    )
    _assert_equal_node_block(
        ig_base.A_col()[ig_base.rank_slices[0], ig_base.rank_slices[0]],
        ig_rewired.A_col()[ig_rewired.rank_slices[0], ig_rewired.rank_slices[0]],
        f"graph {graph_idx} CellCycleLifting (base vs rewired, A_col node-node)",
    )
    print(f"  graph {graph_idx}: CellCycleLifting rewired == base (OK, {ig_base.rank_sizes[1]} edges)")


def check_hypergraph_rewired_equals_complex(lifting, lifting_name: str, data, graph_idx: int) -> None:
    levels = build_influence_graph_levels(lifting, data.clone())
    ig_rewired, ig_complex = levels["rewired"], levels["complex"]

    assert ig_rewired.rank_sizes == ig_complex.rank_sizes, (
        f"graph {graph_idx} {lifting_name}: rewired has ranks {ig_rewired.rank_sizes} "
        f"!= complex {ig_complex.rank_sizes} -- expected identical (ranks 0/1 only)"
    )
    diff = (ig_rewired.A_tilde - ig_complex.A_tilde)
    max_abs_diff = np.abs(diff.toarray()).max() if diff.nnz else 0.0
    assert max_abs_diff < 1e-8, (
        f"graph {graph_idx} {lifting_name}: rewired != complex (max |diff| Ã = {max_abs_diff})"
    )
    print(f"  graph {graph_idx}: {lifting_name} rewired == complex (OK, expected no-op)")


def main() -> None:
    dataset = TUDataset(root=DATA_ROOT, name="MUTAG")
    print(f"MUTAG loaded: {len(dataset)} graphs")

    hyper_liftings = {
        "HypergraphKHopLifting": HypergraphKHopLifting(k_value=1),
        "HypergraphKNNLifting": HypergraphKNNLifting(k_value=3),
        "HypergraphKernelLifting": HypergraphKernelLifting(t=5.0),
    }

    print("\n=== CellCycleLifting: rewired == base (1-skeleton preserved) ===")
    for i in range(min(N_GRAPHS, len(dataset))):
        check_cell_cycle_preserves_1_skeleton(dataset[i], i)

    for name, lifting in hyper_liftings.items():
        print(f"\n=== {name}: rewired == complex (no-op truncation) ===")
        for i in range(min(N_GRAPHS, len(dataset))):
            check_hypergraph_rewired_equals_complex(lifting, name, dataset[i], i)

    print("\nAll 3-level ablation checks passed.")


if __name__ == "__main__":
    main()
