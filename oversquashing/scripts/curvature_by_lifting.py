"""Extended Forman-Ricci curvature (EFC), comparison by lifting.

For each (dataset, lifting, graph), builds the "complex" level of the
ablation (cf. `lift_adapter.build_influence_graph_levels`) and computes, on
its `A_col()`:

1. **EFC distribution** (descriptive, per-graph mean/std/frac-negative,
   aggregated per lifting) -- NOT used alone to claim "lifting X reduces
   oversquashing" (a rightward shift can be a pure artifact of adding many
   cells in dense regions -- use the size-robust wc/nwc summary below instead).
2. **wc / nwc** (betweenness-weighted curvature, global but size-robust)
   -- the equitable global summary to actually compare liftings.
3. **Paired EFC** (before/after on shared base-graph edges) -- only defined
   for liftings that preserve the original 1-skeleton (`CellCycleLifting`,
   `GNN_brut` trivially); hypergraph liftings replace edges with hyperedges
   entirely, so pairing breaks (skipped here, use ΔS_highorder/d_inf from
   the sensitivity comparison instead for those).

∂lift excluded (stochastic, separate trained-model pipeline) -- same static
baselines as the other structural comparisons.

Outputs:
- analysis/curvature/<dataset>_efc_distribution_by_lifting.csv
- analysis/curvature/<dataset>_wc_nwc_by_lifting.csv
- analysis/curvature/efc_paired_<lifting>_<dataset>.csv (CellCycleLifting, GNN_brut)

Usage: `python -m oversquashing.scripts.curvature_by_lifting --dataset MUTAG`
(from the repo root, in the `difflifting` conda env).
"""

import argparse
import os
import os.path as osp

import numpy as np
import pandas as pd
from torch_geometric.datasets import TUDataset

from oversquashing.curvature import (
    betweenness_weighted_curvature,
    curvature_summary,
    extended_forman_curvature,
    paired_efc_on_shared_edges,
)
from oversquashing.lift_adapter import build_influence_graph_levels, build_influence_graph_raw
from oversquashing.scripts.cheeger_by_lifting import get_liftings

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DATA_ROOT = osp.join(REPO_ROOT, "DATA", "DATASETS")
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")

PAIRED_LIFTINGS = {"GNN_brut", "CellCycleLifting"}  # 1-skeleton-preserving liftings only


def _base_edges(data) -> list:
    """Undirected, deduplicated, self-loop-free edge list (node index pairs)
    of the original graph -- same convention as `lift_adapter.raw_graph_incidence`."""
    edge_index = data.edge_index.detach().cpu().numpy()
    return sorted({
        (int(u), int(v)) if u < v else (int(v), int(u))
        for u, v in zip(edge_index[0], edge_index[1])
        if u != v
    })


def _levels_for(lifting_name: str, lifting, data) -> dict:
    if lifting_name == "GNN_brut":
        ig = build_influence_graph_raw(data.clone())
        return {"base": ig, "rewired": ig, "complex": ig}
    return build_influence_graph_levels(lifting, data.clone())


def run_dataset(dataset_name: str, max_graphs: int | None) -> None:
    dataset = TUDataset(root=osp.join(DATA_ROOT, dataset_name), name=dataset_name)
    n_graphs = len(dataset) if max_graphs is None else min(max_graphs, len(dataset))
    print(f"{dataset_name}: using {n_graphs}/{len(dataset)} graphs")

    distribution_rows: dict[str, list] = {}
    wc_rows: dict[str, list] = {}
    paired_rows: dict[str, list] = {}

    liftings = get_liftings()
    for lifting_name, lifting in liftings.items():
        n_failed = 0
        for i in range(n_graphs):
            data = dataset[i]
            try:
                levels = _levels_for(lifting_name, lifting, data)
                ig_complex = levels["complex"]
                A_complex = ig_complex.A_col()
                efc_complex = extended_forman_curvature(A_complex)

                distribution_rows.setdefault(lifting_name, []).append(curvature_summary(efc_complex))
                wc_rows.setdefault(lifting_name, []).append(betweenness_weighted_curvature(A_complex, efc_complex))

                if lifting_name in PAIRED_LIFTINGS:
                    ig_base = levels["base"]
                    efc_base = extended_forman_curvature(ig_base.A_col())
                    edges = _base_edges(data)
                    paired_rows.setdefault(lifting_name, []).extend(
                        paired_efc_on_shared_edges(efc_base, efc_complex, edges)
                    )
            except Exception:
                n_failed += 1
                continue
        if n_failed:
            print(f"  [{lifting_name}] {n_failed}/{n_graphs} graphs failed (skipped)")
        print(f"  [{lifting_name}] done ({n_graphs - n_failed}/{n_graphs} graphs usable)")

    _write_distribution_csv(dataset_name, distribution_rows)
    _write_wc_csv(dataset_name, wc_rows)
    _write_paired_csv(dataset_name, paired_rows)


def _write_distribution_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for lifting_name, summaries in rows.items():
        n_edges = np.array([s["n_edges"] for s in summaries], dtype=float)
        efc_mean = np.array([s["efc_mean"] for s in summaries], dtype=float)
        efc_std = np.array([s["efc_std"] for s in summaries], dtype=float)
        frac_neg = np.array([s["frac_negative"] for s in summaries], dtype=float)
        out.append({
            "dataset": dataset_name,
            "lifting": lifting_name,
            "n_graphs": len(summaries),
            "n_edges_mean": float(np.nanmean(n_edges)),
            "efc_mean_mean": float(np.nanmean(efc_mean)),
            "efc_mean_std": float(np.nanstd(efc_mean)),
            "frac_negative_mean": float(np.nanmean(frac_neg)),
        })
    df = pd.DataFrame(out).sort_values("lifting")
    curvature_dir = osp.join(ANALYSIS_ROOT, "curvature")
    os.makedirs(curvature_dir, exist_ok=True)
    path = osp.join(curvature_dir, f"{dataset_name}_efc_distribution_by_lifting.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _write_wc_csv(dataset_name: str, rows: dict) -> None:
    out = []
    for lifting_name, values in rows.items():
        wc = np.array([v["wc"] for v in values], dtype=float)
        nwc = np.array([v["nwc"] for v in values], dtype=float)
        out.append({
            "dataset": dataset_name,
            "lifting": lifting_name,
            "n_graphs": len(values),
            "wc_mean": float(np.nanmean(wc)),
            "wc_std": float(np.nanstd(wc)),
            "nwc_mean": float(np.nanmean(nwc)),
            "nwc_std": float(np.nanstd(nwc)),
        })
    df = pd.DataFrame(out).sort_values("lifting")
    curvature_dir = osp.join(ANALYSIS_ROOT, "curvature")
    os.makedirs(curvature_dir, exist_ok=True)
    path = osp.join(curvature_dir, f"{dataset_name}_wc_nwc_by_lifting.csv")
    df.to_csv(path, index=False)
    print(f"CSV written: {path}")


def _write_paired_csv(dataset_name: str, rows: dict) -> None:
    curvature_dir = osp.join(ANALYSIS_ROOT, "curvature")
    os.makedirs(curvature_dir, exist_ok=True)
    for lifting_name, pairs in rows.items():
        df = pd.DataFrame(pairs)
        path = osp.join(curvature_dir, f"efc_paired_{lifting_name}_{dataset_name}.csv")
        df.to_csv(path, index=False)
        print(f"CSV written: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--max_graphs", type=int, default=None)
    args = parser.parse_args()
    run_dataset(args.dataset, max_graphs=args.max_graphs)
