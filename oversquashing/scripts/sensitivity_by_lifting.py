"""Structural sensitivity (B^t), comparison per lifting.

For each (dataset, lifting, graph), builds the 3-level ablation
(base/rewired/complex, cf. `lift_adapter.build_influence_graph_levels`) and
computes:

1. **d_inf** (influence distance) on the "rewired" and "complex" levels,
   restricted to the node-node block, bucketed by
   base geodesic distance -> main figure: d_inf vs. geodesic distance, one
   curve per lifting (at the "complex" level).
2. **ΔS_rewire = S_rewired - S_base** and **ΔS_highorder = S_complex - S_rewired**
   (S = (B_hat)^t restricted to the node-node block), swept over t=1..t_max,
   bucketed by base geodesic distance.
3. **0<->2 vertical block** (`CellCycleLifting` only, the only lifting in
   scope with rank 2): (B_hat^t)_{face,node} / (B_hat^t)_{node,face}
   for t=1,2,3 -- quantifies the cost of the vertical detour (zero at t=1,
   activates at t=2 via the edges).

Structural only here (no crossing with the empirical readout, that's a
separate, later check). ∂lift excluded (stochastic, separate trained-model
pipeline) -- same static baselines as the Cheeger/spectral comparison.

Outputs:
- analysis/sensitivity/<dataset>_dinf_by_lifting.csv
- analysis/sensitivity/<dataset>_deltaS_nodenode_by_lifting.csv
- analysis/sensitivity/<dataset>_vertical_0to2_cellcycle.csv
- analysis/figures/<dataset>_dinf_vs_geodesic.png
- analysis/figures/<dataset>_example<i>_St_heatmap_by_rank.png
- analysis/figures/<dataset>_example<i>_deltaS_highorder_nodenode.png

Usage: `python -m oversquashing.scripts.sensitivity_by_lifting --dataset MUTAG`
(from the repo root, in the `difflifting` conda env).
"""

import argparse
import os
import os.path as osp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset

from oversquashing.influence_graph import InfluenceGraph
from oversquashing.lift_adapter import build_influence_graph_levels, build_influence_graph_raw
from oversquashing.scripts.cheeger_by_lifting import get_liftings
from oversquashing.sensitivity import (
    base_geodesic_distance,
    bt_matrix_sequence,
    influence_distance,
    node_block,
    normalized_B,
)

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DATA_ROOT = osp.join(REPO_ROOT, "DATA", "DATASETS")
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")

T_MAX_DEFAULT = 6
N_EXAMPLES_DEFAULT = 2


def _levels_for(lifting_name: str, lifting, data) -> dict:
    """GNN_brut has no lifting: the ablation's 3 levels all coincide with
    the base graph (nothing to rewire or add)."""
    if lifting_name == "GNN_brut":
        ig = build_influence_graph_raw(data.clone())
        return {"base": ig, "rewired": ig, "complex": ig}
    return build_influence_graph_levels(lifting, data.clone())


def _bucket_by_distance(values: np.ndarray, distances: np.ndarray) -> dict:
    """Groups the off-diagonal entries of `values` (n x n) by the
    corresponding integer base geodesic distance (n x n, `np.inf` if not
    connected -- ignored). Returns {distance_int: array_of_values}."""
    n = values.shape[0]
    mask_offdiag = ~np.eye(n, dtype=bool)
    finite = np.isfinite(distances) & mask_offdiag
    buckets: dict[int, list] = {}
    for d in np.unique(distances[finite]).astype(int):
        sel = finite & (distances == d)
        buckets.setdefault(int(d), []).append(values[sel])
    return {d: np.concatenate(v) for d, v in buckets.items()}


def run_dataset(dataset_name: str, max_graphs: int | None, t_max: int, n_examples: int) -> None:
    dataset = TUDataset(root=osp.join(DATA_ROOT, dataset_name), name=dataset_name)
    n_graphs = len(dataset) if max_graphs is None else min(max_graphs, len(dataset))
    print(f"{dataset_name}: using {n_graphs}/{len(dataset)} graphs")

    dinf_rows: dict[tuple, list] = {}
    delta_rows: dict[tuple, list] = {}
    vertical_rows: dict[tuple, list] = {}

    example_graph_idx = [i for i in range(n_graphs) if dataset[i].num_nodes >= 4][:n_examples]

    liftings = get_liftings()
    for lifting_name, lifting in liftings.items():
        n_failed = 0
        for i in range(n_graphs):
            data = dataset[i]
            try:
                levels = _levels_for(lifting_name, lifting, data)
                ig_base, ig_rewired, ig_complex = levels["base"], levels["rewired"], levels["complex"]

                dist_base = base_geodesic_distance(ig_base)

                dinf_rewired = node_block(influence_distance(ig_rewired), ig_rewired)
                dinf_complex = node_block(influence_distance(ig_complex), ig_complex)
                for level_name, dinf in [("rewired", dinf_rewired), ("complex", dinf_complex)]:
                    for d, vals in _bucket_by_distance(dinf, dist_base).items():
                        vals = vals[np.isfinite(vals)]
                        if vals.size:
                            dinf_rows.setdefault((lifting_name, level_name, d), []).append(vals)

                B_base = bt_matrix_sequence(normalized_B(ig_base), t_max)
                B_rewired = bt_matrix_sequence(normalized_B(ig_rewired), t_max)
                B_complex = bt_matrix_sequence(normalized_B(ig_complex), t_max)
                for t in range(1, t_max + 1):
                    S_base = node_block(B_base[t - 1], ig_base)
                    S_rewired = node_block(B_rewired[t - 1], ig_rewired)
                    S_complex = node_block(B_complex[t - 1], ig_complex)
                    delta_rewire = S_rewired - S_base
                    delta_highorder = S_complex - S_rewired
                    for delta_type, delta in [("rewire", delta_rewire), ("highorder", delta_highorder)]:
                        for d, vals in _bucket_by_distance(delta, dist_base).items():
                            delta_rows.setdefault((lifting_name, delta_type, t, d), []).append(vals)

                if lifting_name == "CellCycleLifting" and len(ig_complex.rank_sizes) > 2:
                    n_node, n_edge, n_face = ig_complex.rank_sizes[:3]
                    if n_face > 0:
                        for t in (1, 2, 3):
                            if t > t_max:
                                break
                            S = B_complex[t - 1]
                            face_slice, node_slice = ig_complex.rank_slices[2], ig_complex.rank_slices[0]
                            face_to_node = S[face_slice, node_slice]
                            node_to_face = S[node_slice, face_slice]
                            for direction, block in [("face_to_node", face_to_node), ("node_to_face", node_to_face)]:
                                vertical_rows.setdefault((t, direction), []).append(
                                    (float(block.mean()), float((block > 0).mean()))
                                )

                if i in example_graph_idx and lifting_name == "CellCycleLifting":
                    t_example = min(3, t_max)
                    _plot_examples(
                        dataset_name, i, ig_base, ig_rewired, ig_complex,
                        B_rewired[t_example - 1], B_complex[t_example - 1],
                    )
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                continue
        if n_failed:
            print(f"  [{lifting_name}] {n_failed}/{n_graphs} graphs failed (skipped)")
        print(f"  [{lifting_name}] done ({n_graphs - n_failed}/{n_graphs} graphs usable)")

    _write_dinf_csv(dataset_name, dinf_rows)
    _write_delta_csv(dataset_name, delta_rows)
    _write_vertical_csv(dataset_name, vertical_rows)
    _plot_dinf_vs_geodesic(dataset_name)


def _write_dinf_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for (lifting_name, level_name, d), chunks in rows.items():
        vals = np.concatenate(chunks)
        out.append({
            "dataset": dataset_name,
            "lifting": lifting_name,
            "level": level_name,
            "geodesic_distance": d,
            "n_pairs": vals.size,
            "dinf_mean": float(vals.mean()),
            "dinf_std": float(vals.std()),
        })
    df = pd.DataFrame(out).sort_values(["lifting", "level", "geodesic_distance"])
    struct_dir = osp.join(ANALYSIS_ROOT, "sensitivity")
    os.makedirs(struct_dir, exist_ok=True)
    path = osp.join(struct_dir, f"{dataset_name}_dinf_by_lifting.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _write_delta_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for (lifting_name, delta_type, t, d), chunks in rows.items():
        vals = np.concatenate(chunks)
        out.append({
            "dataset": dataset_name,
            "lifting": lifting_name,
            "delta_type": delta_type,
            "t": t,
            "geodesic_distance": d,
            "n_pairs": vals.size,
            "delta_mean": float(vals.mean()),
            "delta_std": float(vals.std()),
        })
    df = pd.DataFrame(out).sort_values(["lifting", "delta_type", "t", "geodesic_distance"])
    struct_dir = osp.join(ANALYSIS_ROOT, "sensitivity")
    os.makedirs(struct_dir, exist_ok=True)
    path = osp.join(struct_dir, f"{dataset_name}_deltaS_nodenode_by_lifting.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _write_vertical_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for (t, direction), samples in rows.items():
        means, fracs = zip(*samples, strict=False)
        out.append({
            "dataset": dataset_name,
            "t": t,
            "direction": direction,
            "n_graphs": len(samples),
            "mean_value_mean": float(np.mean(means)),
            "frac_nonzero_mean": float(np.mean(fracs)),
        })
    df = pd.DataFrame(out).sort_values(["direction", "t"])
    struct_dir = osp.join(ANALYSIS_ROOT, "sensitivity")
    os.makedirs(struct_dir, exist_ok=True)
    path = osp.join(struct_dir, f"{dataset_name}_vertical_0to2_cellcycle.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _plot_dinf_vs_geodesic(dataset_name: str) -> None:
    path = osp.join(ANALYSIS_ROOT, "sensitivity", f"{dataset_name}_dinf_by_lifting.csv")
    df = pd.read_csv(path)
    df = df[df["level"] == "complex"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for lifting_name, sub in df.groupby("lifting"):
        sub = sub.sort_values("geodesic_distance")
        ax.plot(sub["geodesic_distance"], sub["dinf_mean"], marker="o", label=lifting_name)
    ax.set_xlabel("base geodesic distance (node-node)")
    ax.set_ylabel("mean d_inf (complex level)")
    ax.set_title(f"{dataset_name}: influence distance vs. base geodesic distance")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = osp.join(figures_dir, f"{dataset_name}_dinf_vs_geodesic.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {fig_path}")


def _plot_examples(
    dataset_name: str,
    graph_idx: int,
    ig_base: InfluenceGraph,
    ig_rewired: InfluenceGraph,
    ig_complex: InfluenceGraph,
    S_rewired: np.ndarray,
    S_complex: np.ndarray,
) -> None:
    """Heatmaps for one example graph, `CellCycleLifting` only (the only
    lifting in scope with rank 2, cf. the module's docstring)."""
    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    # (1) full heatmap of S=(B_hat)^t (complex level), ordered by rank.
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.log1p(S_complex), cmap="viridis")
    for offset in ig_complex.rank_slices[1:]:
        ax.axhline(offset.start - 0.5, color="white", linewidth=0.8)
        ax.axvline(offset.start - 0.5, color="white", linewidth=0.8)
    ax.set_title(f"{dataset_name} graph {graph_idx}: log(1+S) CellCycleLifting, by rank")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, f"{dataset_name}_example{graph_idx}_St_heatmap_by_rank.png"), dpi=150)
    plt.close(fig)

    # (2) DeltaS_highorder = S_complex - S_rewired, node-node block, nodes
    # ordered by base geodesic distance from node 0.
    delta_highorder = node_block(S_complex, ig_complex) - node_block(S_rewired, ig_rewired)
    dist_base = base_geodesic_distance(ig_base)
    order = np.argsort(np.where(np.isfinite(dist_base[0]), dist_base[0], np.inf))

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(delta_highorder[np.ix_(order, order)], cmap="coolwarm")
    ax.set_title(
        f"{dataset_name} graph {graph_idx}: deltaS_highorder node-node,\n"
        "sorted by geodesic distance from node 0"
    )
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(osp.join(figures_dir, f"{dataset_name}_example{graph_idx}_deltaS_highorder_nodenode.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--max_graphs", type=int, default=None)
    parser.add_argument("--t_max", type=int, default=T_MAX_DEFAULT)
    parser.add_argument("--n_examples", type=int, default=N_EXAMPLES_DEFAULT)
    args = parser.parse_args()
    run_dataset(args.dataset, max_graphs=args.max_graphs, t_max=args.t_max, n_examples=args.n_examples)
