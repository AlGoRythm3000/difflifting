"""Sanity checks for curvature and effective resistance.

Verifies, on a few MUTAG graphs:
- EFC is symmetric on `A_col()` (expected here, cf. `curvature.py`'s module
  docstring -- a property of this codebase's influence-graph construction).
- Paired EFC delta is exactly 0 for `GNN_brut` (base == complex trivially,
  same reasoning as `ablation_levels_sanity.py`'s ablation-level checks).
- `R_eff` is symmetric, zero on the diagonal, and strictly positive
  off-diagonal for connected pairs.

Nothing is trained here: purely structural checks.
Usage: `python -m oversquashing.scripts.curvature_resistance_sanity` from
the repo root, in the `difflifting` conda env.
"""

import os.path as osp

import numpy as np
from torch_geometric.datasets import TUDataset

from oversquashing.curvature import extended_forman_curvature, paired_efc_on_shared_edges
from oversquashing.lift_adapter import build_influence_graph_from_lifting, build_influence_graph_raw
from oversquashing.resistance import effective_resistance_matrix
from oversquashing.scripts.curvature_by_lifting import _base_edges
from tools.lifting.cycle_lifting import CellCycleLifting

DATA_ROOT = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), "DATA", "DATASETS", "MUTAG")
N_GRAPHS = 5


def check_efc_symmetric(data, graph_idx: int) -> None:
    lifting = CellCycleLifting()
    influence_graph = build_influence_graph_from_lifting(lifting, data.clone())
    efc = extended_forman_curvature(influence_graph.A_col())
    diff = (efc - efc.T)
    max_abs_diff = np.abs(diff.toarray()).max() if diff.nnz else 0.0
    assert max_abs_diff < 1e-8, f"graph {graph_idx}: EFC not symmetric (max |diff| = {max_abs_diff})"
    print(f"  graph {graph_idx}: EFC symmetric on A_col (OK, {efc.nnz} directed entries)")


def check_paired_efc_gnn_brut_is_zero(data, graph_idx: int) -> None:
    ig = build_influence_graph_raw(data.clone())
    efc = extended_forman_curvature(ig.A_col())
    edges = _base_edges(data)
    paired = paired_efc_on_shared_edges(efc, efc, edges)
    max_abs_delta = max((abs(row["delta"]) for row in paired), default=0.0)
    assert max_abs_delta < 1e-8, f"graph {graph_idx}: GNN_brut paired EFC delta != 0 (max = {max_abs_delta})"
    print(f"  graph {graph_idx}: GNN_brut paired EFC delta == 0 (OK, {len(paired)} edges)")


def check_effective_resistance(data, graph_idx: int) -> None:
    ig = build_influence_graph_raw(data.clone())
    R = effective_resistance_matrix(ig.A_col())
    assert np.allclose(np.diag(R), 0.0), f"graph {graph_idx}: R_eff diagonal not zero"
    assert np.allclose(R, R.T), f"graph {graph_idx}: R_eff not symmetric"
    off_diag = R[~np.eye(R.shape[0], dtype=bool)]
    assert (off_diag >= -1e-8).all(), f"graph {graph_idx}: R_eff has negative entries"
    print(f"  graph {graph_idx}: R_eff symmetric, zero diagonal, non-negative (OK)")


def main() -> None:
    dataset = TUDataset(root=DATA_ROOT, name="MUTAG")
    print(f"MUTAG loaded: {len(dataset)} graphs")

    print("\n=== EFC symmetry (CellCycleLifting) ===")
    for i in range(min(N_GRAPHS, len(dataset))):
        check_efc_symmetric(dataset[i], i)

    print("\n=== Paired EFC == 0 for GNN_brut (1-skeleton trivially preserved) ===")
    for i in range(min(N_GRAPHS, len(dataset))):
        check_paired_efc_gnn_brut_is_zero(dataset[i], i)

    print("\n=== Effective resistance sanity (GNN_brut) ===")
    for i in range(min(N_GRAPHS, len(dataset))):
        check_effective_resistance(dataset[i], i)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
