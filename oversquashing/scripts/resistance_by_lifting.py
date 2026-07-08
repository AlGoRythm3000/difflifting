"""Effective resistance between fixed base-node pairs, by lifting.

For each (dataset, lifting, graph), builds the "complex" level of the
ablation (cf. `lift_adapter.build_influence_graph_levels`) and computes
effective resistance `R_eff(u,v)` (`resistance.effective_resistance_matrix`)
between every pair of *base* nodes, over the full lifted structure
(higher-rank cells act as extra conductors) -- the primary size-robust,
apples-to-apples comparison across liftings: same node pairs, different
topology in between, insensitive to how many cells a lifting adds.

Bucketed by base geodesic distance (same convention as
`sensitivity.base_geodesic_distance`, reused from `sensitivity_by_lifting.py`)
so the comparison reads as a curve, not a single scalar (same "équité" logic
as d_inf in the sensitivity comparison: a lifting that lowers R_eff only for
already-close pairs is not evidence of reduced oversquashing).

∂lift excluded (stochastic, separate trained-model pipeline) -- same static
baselines as the other structural comparisons.

Outputs:
- analysis/fair/<dataset>_reff_by_lifting.csv
- analysis/figures/<dataset>_reff_vs_geodesic.png

Usage: `python -m oversquashing.scripts.resistance_by_lifting --dataset MUTAG`
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

from oversquashing.lift_adapter import build_influence_graph_levels, build_influence_graph_raw
from oversquashing.resistance import effective_resistance_matrix
from oversquashing.scripts.cheeger_by_lifting import get_liftings
from oversquashing.scripts.sensitivity_by_lifting import _bucket_by_distance
from oversquashing.sensitivity import base_geodesic_distance, node_block

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DATA_ROOT = osp.join(REPO_ROOT, "DATA", "DATASETS")
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")


def _levels_for(lifting_name: str, lifting, data) -> dict:
    if lifting_name == "GNN_brut":
        ig = build_influence_graph_raw(data.clone())
        return {"base": ig, "rewired": ig, "complex": ig}
    return build_influence_graph_levels(lifting, data.clone())


def run_dataset(dataset_name: str, max_graphs: int | None) -> None:
    dataset = TUDataset(root=osp.join(DATA_ROOT, dataset_name), name=dataset_name)
    n_graphs = len(dataset) if max_graphs is None else min(max_graphs, len(dataset))
    print(f"{dataset_name}: using {n_graphs}/{len(dataset)} graphs")

    reff_rows: dict[tuple, list] = {}

    liftings = get_liftings()
    for lifting_name, lifting in liftings.items():
        n_failed = 0
        for i in range(n_graphs):
            data = dataset[i]
            try:
                levels = _levels_for(lifting_name, lifting, data)
                ig_base, ig_complex = levels["base"], levels["complex"]

                dist_base = base_geodesic_distance(ig_base)
                reff_full = effective_resistance_matrix(ig_complex.A_col())
                reff_nodes = node_block(reff_full, ig_complex)

                for d, vals in _bucket_by_distance(reff_nodes, dist_base).items():
                    reff_rows.setdefault((lifting_name, d), []).append(vals)
            except Exception:
                n_failed += 1
                continue
        if n_failed:
            print(f"  [{lifting_name}] {n_failed}/{n_graphs} graphs failed (skipped)")
        print(f"  [{lifting_name}] done ({n_graphs - n_failed}/{n_graphs} graphs usable)")

    _write_csv(dataset_name, reff_rows)
    _plot(dataset_name)


def _write_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for (lifting_name, d), chunks in rows.items():
        vals = np.concatenate(chunks)
        out.append({
            "dataset": dataset_name,
            "lifting": lifting_name,
            "geodesic_distance": d,
            "n_pairs": vals.size,
            "reff_mean": float(vals.mean()),
            "reff_std": float(vals.std()),
        })
    df = pd.DataFrame(out).sort_values(["lifting", "geodesic_distance"])
    fair_dir = osp.join(ANALYSIS_ROOT, "fair")
    os.makedirs(fair_dir, exist_ok=True)
    path = osp.join(fair_dir, f"{dataset_name}_reff_by_lifting.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _plot(dataset_name: str) -> None:
    path = osp.join(ANALYSIS_ROOT, "fair", f"{dataset_name}_reff_by_lifting.csv")
    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(7, 5))
    for lifting_name, sub in df.groupby("lifting"):
        sub = sub.sort_values("geodesic_distance")
        ax.plot(sub["geodesic_distance"], sub["reff_mean"], marker="o", label=lifting_name)
    ax.set_xlabel("base geodesic distance (node-node)")
    ax.set_ylabel("mean effective resistance (complex level)")
    ax.set_title(f"{dataset_name}: effective resistance vs base geodesic distance")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = osp.join(figures_dir, f"{dataset_name}_reff_vs_geodesic.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()
    run_dataset(args.dataset, max_graphs=args.max_graphs)
