"""Cheeger / spectral gap, comparison per lifting and per scope.

For a dataset (MUTAG then PROTEINS), computes λ₂ + Cheeger bounds + Fiedler
sweep on the 4 scopes
(global/vertical/horizontal/horizontal-per-rank) of the influence graph of
each static lifting, averaged over every graph in the dataset.

Liftings covered : GNN_brut (1-skeleton reference,
no lifting), CellCycleLifting (cellular), HypergraphKHopLifting,
HypergraphKNNLifting, HypergraphKernelLifting. ∂lift is deliberately
excluded from this pass: it's stochastic and needs the full trained-model
pipeline (average over K=10 samples) -- handled separately.

Note: the "GNN_brut" identifier (French for "raw GNN") is kept as-is here
and throughout every later script/output for consistency with the CSVs,
figures, and notes already produced under that name.

Outputs :
- analysis/struct/<dataset>_lambda2_cheeger.csv
- analysis/figures/<dataset>_lambda2_by_lifting_scope.png

Usage: `python -m oversquashing.scripts.cheeger_by_lifting --dataset MUTAG`
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

from oversquashing.lift_adapter import build_influence_graph_from_lifting, build_influence_graph_raw
from oversquashing.spectral import compute_scope_metrics
from tools.lifting.cycle_lifting import CellCycleLifting
from tools.lifting.hypergraph import HypergraphKHopLifting, HypergraphKNNLifting
from tools.lifting.kernel import HypergraphKernelLifting

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DATA_ROOT = osp.join(REPO_ROOT, "DATA", "DATASETS")
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")

METRIC_KEYS = ["n_cells", "n_edges", "n_isolated", "lambda2", "cheeger_lo", "cheeger_hi", "cheeger_sweep"]


def get_liftings() -> dict:
    """Static liftings in scope (§3), + None for the GNN_brut reference
    (handled separately since it doesn't go through `lift_topology`)."""
    return {
        "GNN_brut": None,
        "CellCycleLifting": CellCycleLifting(),
        "HypergraphKHopLifting": HypergraphKHopLifting(k_value=1),
        "HypergraphKNNLifting": HypergraphKNNLifting(k_value=3),
        "HypergraphKernelLifting": HypergraphKernelLifting(t=5.0),
    }


def run_dataset(dataset_name: str, max_graphs: int | None = None) -> pd.DataFrame:
    dataset = TUDataset(root=osp.join(DATA_ROOT, dataset_name), name=dataset_name)
    n_graphs = len(dataset) if max_graphs is None else min(max_graphs, len(dataset))
    print(f"{dataset_name}: using {n_graphs}/{len(dataset)} graphs")

    rows = []
    for lifting_name, lifting in get_liftings().items():
        per_scope_metrics: dict[str, list[dict]] = {}
        n_failed = 0
        for i in range(n_graphs):
            data = dataset[i]
            try:
                if lifting_name == "GNN_brut":
                    ig = build_influence_graph_raw(data.clone())
                else:
                    ig = build_influence_graph_from_lifting(lifting, data.clone())
                metrics = compute_scope_metrics(ig)
            except Exception:
                n_failed += 1
                continue
            for scope, m in metrics.items():
                per_scope_metrics.setdefault(scope, []).append(m)

        if n_failed:
            print(f"  [{lifting_name}] {n_failed}/{n_graphs} graphs failed (skipped, see exceptions)")

        for scope, metric_list in per_scope_metrics.items():
            row = {
                "dataset": dataset_name,
                "lifting": lifting_name,
                "scope": scope,
                "n_graphs": len(metric_list),
            }
            for key in METRIC_KEYS:
                values = np.array([m[key] for m in metric_list], dtype=float)
                row[f"{key}_mean"] = np.nanmean(values)
                row[f"{key}_std"] = np.nanstd(values)
            rows.append(row)
        print(f"  [{lifting_name}] done ({n_graphs - n_failed}/{n_graphs} graphs usable)")

    df = pd.DataFrame(rows)

    struct_dir = osp.join(ANALYSIS_ROOT, "struct")
    os.makedirs(struct_dir, exist_ok=True)
    csv_path = osp.join(struct_dir, f"{dataset_name}_lambda2_cheeger.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV written: {csv_path}")

    plot_lambda2(df, dataset_name)
    return df


def plot_lambda2(df: pd.DataFrame, dataset_name: str) -> None:
    scopes_order = ["global", "vertical", "horizontal"] + sorted(
        s for s in df["scope"].unique() if s.startswith("horizontal_rank")
    )
    liftings = list(df["lifting"].unique())

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(scopes_order)), 5))
    width = 0.8 / max(len(liftings), 1)
    x = np.arange(len(scopes_order))
    for li, lifting_name in enumerate(liftings):
        sub = df[df["lifting"] == lifting_name].set_index("scope")
        means = [sub.loc[s, "lambda2_mean"] if s in sub.index else np.nan for s in scopes_order]
        stds = [sub.loc[s, "lambda2_std"] if s in sub.index else 0.0 for s in scopes_order]
        ax.bar(x + li * width, means, width=width, yerr=stds, label=lifting_name, capsize=2)

    ax.set_xticks(x + width * (len(liftings) - 1) / 2)
    ax.set_xticklabels(scopes_order, rotation=30, ha="right")
    ax.set_ylabel("λ2 (normalized Laplacian L_sym), mean ± std")
    ax.set_title(f"{dataset_name}: λ2 per lifting and per scope")
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = osp.join(figures_dir, f"{dataset_name}_lambda2_by_lifting_scope.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()
    run_dataset(args.dataset, max_graphs=args.max_graphs)
