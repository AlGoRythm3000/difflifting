"""Mechanistic study: does DiffLift's acceptance probability phi(z_C)
correlate with the shortcut a candidate cell C actually creates?

DiffLift optimizes for the task loss, not for oversquashing -- this is a
falsifiable check of the paper's implicit
claim that the task-optimal complex reduces oversquashing between relevant
node pairs.

Protocol (structural, on a trained diffLifting model, deterministic k_v for
reproducibility -- see `model/tnn_with_lifting_graph_classific.py`):
1. Train a `TNN_KNN_MLP_G(diff_lifting=True)` model briefly on MUTAG (cellular
   TNN, e.g. CWN, so both candidate edges (rank 1) and candidate 2-cells
   (rank 2) exist and have exposed acceptance probabilities).
2. Run a forward pass per graph (eval mode); the model already computes
   phi(z_C) for every candidate edge/cell internally -- exposed on `batch` as
   `candidate_edge_phi`/`candidate_edge_pairs`/`candidate_edge_kept` and
   `candidate_cell_phi`/`candidate_cell_kept`/`candidate_cell_nodes` (added as
   read-only instrumentation, does not change the forward pass' behavior).
   Uses `candidate_edge_phi_raw`/`candidate_cell_phi_raw` (pre-sharpening
   probabilities) as phi(z_C) rather than the sharpened `*_phi` fields: an
   early dry run showed the sharpened probabilities saturate to ~1.0 for
   essentially every accepted cell (sharpening_factor=10 in the model makes
   softmax nearly binary), leaving no dynamic range to correlate against
   delta_reff. The raw and sharpened probabilities agree on every accept/
   reject decision (sharpening by a positive factor preserves the 0.5
   threshold) -- only the raw one has enough spread to be informative here.
3. For each *accepted* candidate cell C, compute its marginal effect:
   `delta_reff(C) = R_eff(complex without C) - R_eff(complex with C)`,
   averaged over the base node pairs C spans (the pair (u,v) for an edge
   candidate; all pairs among the cycle's nodes for a 2-cell candidate).
4. Correlate phi(z_C) and delta_reff(C) across all accepted cells (Spearman).

Reading: positive delta_reff means C lowers resistance, i.e. C is a genuine
shortcut. A strong positive phi-vs-delta_reff correlation would support
DiffLift's implicit claim; a null/negative correlation despite good task
accuracy would suggest DiffLift helps for a reason other than oversquashing
reduction (an equally interesting, publishable result either way).

Known pre-existing gap in the model code (not fixed here, out of scope for
this analysis-only pass): `TNN_KNN_MLP_G.__init__` accepts a `deterministic`
argument but never stores it as `self.deterministic`, so `--deterministic`
has no effect via the CLI/constructor. This script sets `model.deterministic
= True` directly as an attribute after construction (Python allows this,
and `forward()` reads it via `getattr`) to get reproducible k_v selection
during the extraction pass.

Output: `analysis/fair/difflift_mechanism_<dataset>.csv` (phi, delta_reff,
kind, graph_idx, n_covered_pairs) + `analysis/figures/<dataset>_difflift_mechanism_scatter.png`.

Usage: `python -m oversquashing.scripts.difflift_mechanism --dataset MUTAG`
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
import torch
from scipy.stats import spearmanr

from dataset.dataset_handler import choose_dataset
from model.tnn_with_lifting_graph_classific import TNN_KNN_MLP_G
from oversquashing.influence_graph import build_influence_graph, to_scipy
from oversquashing.resistance import effective_resistance_matrix
from train import evaluate, train
from utils import set_seed

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")


def build_args(dataset_name: str, tnn_type: str, k_max: int, epochs: int) -> argparse.Namespace:
    """Minimal args namespace covering everything `choose_dataset` /
    `TNN_KNN_MLP_G` reads -- mirrors `utils.parse_args`' defaults, overriding
    only what this script needs (dataset, tnn, lifting, epoch budget)."""
    return argparse.Namespace(
        seed=42, gnn="GIN", tnn=tnn_type, dataset=dataset_name, lifting="diffLifting",
        lr=0.005, weight_decay=0.0, batch_size=1, num_layers=2, num_layers_gnn=2,
        max_epochs=epochs, number_of_mask=1, early_stop_patience=50, lr_decay_patience=10,
        logdir="results/", hidden_dim=32, gnn_embedding_dim=32, k_max=k_max, k=k_max,
        graph_transformer_n_heads=4, positional_encoder_dim=4, positional_walking_len=20,
        depth=2, no_readout=False, signed=False, use_dcm_split=False, bn=True,
        deepset_aggr_type="sum", sub_gccn_model="GIN", global_pooling="mean",
        t=5.0, deterministic=False,
    )


def train_model(dataset_name: str, tnn_type: str, k_max: int, epochs: int, device):
    args = build_args(dataset_name, tnn_type, k_max, epochs)
    set_seed(args.seed)
    (train_loader, val_loader, test_loader), num_features, num_classes = choose_dataset(args, device)

    model = TNN_KNN_MLP_G(
        num_features, args, hidden_dim=args.hidden_dim, num_classes=num_classes,
        k=k_max, diff_lifting=True, global_pool=args.global_pooling, device=device,
        tnn_type=tnn_type, num_layers_tnn=args.num_layers, num_layers_gnn=args.num_layers_gnn,
        embedding_dim=args.gnn_embedding_dim, k_max=k_max, deterministic=False,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")

    for epoch in range(1, epochs + 1):
        train_loss = train(train_loader, model, loss_fn, optimizer, device)
        if epoch % 5 == 0 or epoch == epochs:
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            print(f"  epoch {epoch:3d}: train_loss={train_loss:.3f} val_acc={val_acc:.3f}")

    return model, train_loader, val_loader, test_loader


def extract_candidates(batch) -> list:
    """Accepted candidate cells from an already-forward-passed batch, with
    their column index in `incidence_1`/`incidence_2` (needed for ablation)."""
    edge_phi, edge_pairs, edge_kept = batch["candidate_edge_phi_raw"], batch["candidate_edge_pairs"], batch["candidate_edge_kept"]
    cell_phi, cell_kept, cell_nodes = batch["candidate_cell_phi_raw"], batch["candidate_cell_kept"], batch["candidate_cell_nodes"]

    n_kept_edges = int(edge_kept.sum().item())
    n_orig_edges = batch["incidence_1"].shape[1] - n_kept_edges

    candidates = []
    kept_edge_idx = edge_kept.nonzero(as_tuple=True)[0]
    for rank, raw_idx in enumerate(kept_edge_idx.tolist()):
        u, v = int(edge_pairs[0, raw_idx]), int(edge_pairs[1, raw_idx])
        if u == v:
            continue  # self-loop candidate, not a meaningful node pair
        candidates.append({"kind": "edge", "col": n_orig_edges + rank, "phi": float(edge_phi[raw_idx]), "nodes": (u, v)})

    if cell_phi.numel() > 0:
        kept_cell_idx = cell_kept.nonzero(as_tuple=True)[0]
        for rank, raw_idx in enumerate(kept_cell_idx.tolist()):
            candidates.append({"kind": "cell", "col": rank, "phi": float(cell_phi[raw_idx]), "nodes": tuple(cell_nodes[raw_idx])})
    return candidates


def build_full_ig(batch):
    B1 = to_scipy(batch["incidence_1"])
    B2 = to_scipy(batch["incidence_2"])
    if B2.shape[1] == 0:
        return build_influence_graph([B1])
    return build_influence_graph([B1, B2])


def ablate(batch, candidate: dict):
    """Influence graph with one accepted candidate cell removed."""
    B1 = to_scipy(batch["incidence_1"])
    B2 = to_scipy(batch["incidence_2"])
    if candidate["kind"] == "edge":
        col = candidate["col"]
        keep_cols = [c for c in range(B1.shape[1]) if c != col]
        B1_new = B1[:, keep_cols]
        B2_new = B2[keep_cols, :] if B2.shape[1] > 0 else B2
    else:
        col = candidate["col"]
        B1_new = B1
        B2_new = B2[:, [c for c in range(B2.shape[1]) if c != col]]
    if B2_new.shape[1] == 0:
        return build_influence_graph([B1_new])
    return build_influence_graph([B1_new, B2_new])


def marginal_delta_reff(batch, candidate: dict, n_nodes: int):
    ig_full = build_full_ig(batch)
    ig_without = ablate(batch, candidate)
    R_full = effective_resistance_matrix(ig_full.A_col())[:n_nodes, :n_nodes]
    R_without = effective_resistance_matrix(ig_without.A_col())[:n_nodes, :n_nodes]

    nodes = candidate["nodes"]
    pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]
    deltas = [R_without[u, v] - R_full[u, v] for u, v in pairs]
    return float(np.mean(deltas)), len(pairs)


def run_mechanism_study(dataset_name: str, tnn_type: str, k_max: int, epochs: int, n_graphs: int, device) -> None:
    print(f"Training diffLifting/{tnn_type} on {dataset_name} for {epochs} epochs...")
    model, train_loader, val_loader, test_loader = train_model(dataset_name, tnn_type, k_max, epochs, device)
    model.eval()
    model.deterministic = True  # deterministic k_v for a reproducible extraction pass

    rows = []
    n_done = 0
    for loader in (train_loader, val_loader, test_loader):
        for batch in loader:
            if n_done >= n_graphs:
                break
            batch = batch.to(device)
            with torch.no_grad():
                model(batch)
            candidates = extract_candidates(batch)
            n_nodes = batch["x"].shape[0]
            for c in candidates:
                delta_reff, n_pairs = marginal_delta_reff(batch, c, n_nodes)
                rows.append({
                    "dataset": dataset_name, "graph_idx": n_done, "kind": c["kind"],
                    "phi": c["phi"], "delta_reff": delta_reff, "n_covered_pairs": n_pairs,
                })
            n_done += 1
        if n_done >= n_graphs:
            break
    print(f"Processed {n_done} graphs, {len(rows)} accepted candidate cells total.")

    df = pd.DataFrame(rows)
    fair_dir = osp.join(ANALYSIS_ROOT, "fair")
    os.makedirs(fair_dir, exist_ok=True)
    csv_path = osp.join(fair_dir, f"difflift_mechanism_{dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV written: {csv_path}")

    if len(df) < 3:
        print("Too few candidate cells for a meaningful correlation, stopping.")
        return

    rho, pval = spearmanr(df["phi"], df["delta_reff"])
    print(f"Spearman(phi, delta_reff) = {rho:.3f} (p={pval:.3g}), n={len(df)}")
    for kind, sub in df.groupby("kind"):
        if len(sub) >= 3:
            rho_k, pval_k = spearmanr(sub["phi"], sub["delta_reff"])
            print(f"  [{kind}] Spearman = {rho_k:.3f} (p={pval_k:.3g}), n={len(sub)}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for kind, sub in df.groupby("kind"):
        ax.scatter(sub["phi"], sub["delta_reff"], label=kind, alpha=0.6, s=15)
    ax.set_xlabel("phi(z_C): acceptance probability")
    ax.set_ylabel("delta_reff(C): R_eff(without C) - R_eff(with C)")
    ax.set_title(f"{dataset_name}: DiffLift acceptance vs. marginal shortcut effect\nSpearman={rho:.3f} (p={pval:.3g}, n={len(df)})")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()

    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = osp.join(figures_dir, f"{dataset_name}_difflift_mechanism_scatter.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--tnn", type=str, default="CWN", help="Cellular TNN type (needs rank-2 cells).")
    parser.add_argument("--k_max", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n_graphs", type=int, default=40, help="Number of graphs to run the mechanistic extraction on.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_mechanism_study(args.dataset, args.tnn, args.k_max, args.epochs, args.n_graphs, device)
