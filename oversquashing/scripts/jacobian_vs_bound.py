"""Empirical Jacobian vs. the structural (B^t) bound (Lemma 3.2).

The bound `||d h_sigma^(t) / d h_tau^(0)||_1 <= const * (B^t)_{sigma,tau}`
is purely structural -- every other check in this module builds on the
influence graph alone, never training or running a real forward pass. This
is the first and only check against a *real* model's autograd Jacobian, on
a fixed (untrained) random init ("fixed init first"; a trained-model check
is future work, not implemented here).

Model used: `model.TNN.CWN` directly (not the full `TNN_KNN_MLP_G`/diffLifting
pipeline -- this is a structural-vs-real-gradient check on a *static* lifting,
independent of DiffLift). Architectural property worth knowing before reading
results: this repo's `CWN` implementation only updates edge (rank-1)
representations across layers (`CWN.forward`'s loop only reassigns `x_1`) --
`x_0`/`x_2` stay at their initial per-cell projection for every layer, no
cross-cell propagation into or out of them beyond that single projection.
So the only *depth-dependent* sensitivity this specific architecture can
exhibit is edges-as-target: `sigma` (probed cells) is restricted to edges;
`tau` (source cells) ranges over nodes/edges/faces. This is a property of
the existing model code, not a simplification introduced here.

Initial node/edge/face features (x_0/x_1/x_2) are independent random tensors
(not derived from each other via feature lifting) so that d h_tau^(0) probes
are genuinely independent leaves, matching the relational framework's
"each cell has its own input signal" setup rather than conflating
feature-lifting propagation with the model's own message passing.

Output: `analysis/sensitivity/<dataset>_jacobian_vs_bound_graph<idx>.csv`
(sigma, tau, tau_rank, t, s_emp, bt_bound) + a log-log scatter figure.

Usage: `python -m oversquashing.scripts.jacobian_vs_bound --dataset MUTAG`
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
from torch_geometric.datasets import TUDataset

from model.TNN import CWN
from oversquashing.lift_adapter import build_influence_graph_from_lifting
from oversquashing.sensitivity import bt_matrix
from tools.lifting.cycle_lifting import CellCycleLifting

REPO_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
DATA_ROOT = osp.join(REPO_ROOT, "DATA", "DATASETS")
ANALYSIS_ROOT = osp.join(REPO_ROOT, "analysis")


def find_graph_with_faces(dataset, min_faces=2, max_scan=50):
    """First graph (by index) whose CellCycleLifting complex has at least
    `min_faces` 2-cells -- otherwise the rank-2 block would be empty/trivial
    and there would be nothing depth-dependent to probe beyond rank 1."""
    lifting = CellCycleLifting()
    for i in range(min(max_scan, len(dataset))):
        ig = build_influence_graph_from_lifting(lifting, dataset[i].clone())
        if len(ig.rank_sizes) > 2 and ig.rank_sizes[2] >= min_faces:
            return i, ig
    raise RuntimeError(f"No graph with >= {min_faces} faces found in the first {max_scan} graphs.")


def empirical_sensitivity(model, x0, x1, x2, adjacency_1, incidence_2, incidence_1_t, n_probe_edges: int):
    """S_emp(sigma, tau) for sigma in the first `n_probe_edges` edges, tau over
    all cells (nodes+edges+faces): entrywise L1 norm of the Jacobian block
    d x1_out[sigma] / d x_tau, summed over both the output and input feature
    dimensions."""
    x0_out, x1_out, x2_out = model(x0, x1, x2, adjacency_1, incidence_2, incidence_1_t)
    hidden_dim = x1_out.shape[1]
    n0, n1, n2 = x0.shape[0], x1.shape[0], x2.shape[0]

    S = np.zeros((n_probe_edges, n0 + n1 + n2))
    for sigma in range(n_probe_edges):
        row = np.zeros(n0 + n1 + n2)
        for d in range(hidden_dim):
            g0, g1, g2 = torch.autograd.grad(x1_out[sigma, d], [x0, x1, x2], retain_graph=True, allow_unused=True)
            if g0 is not None:
                row[:n0] += g0.abs().sum(dim=1).detach().numpy()
            if g1 is not None:
                row[n0:n0 + n1] += g1.abs().sum(dim=1).detach().numpy()
            if g2 is not None:
                row[n0 + n1:] += g2.abs().sum(dim=1).detach().numpy()
        S[sigma] = row
    return S


def run(dataset_name: str, hidden_dim: int, t_values, n_probe_edges: int, seed: int) -> None:
    torch.manual_seed(seed)
    dataset = TUDataset(root=osp.join(DATA_ROOT, dataset_name), name=dataset_name)
    graph_idx, ig = find_graph_with_faces(dataset)
    print(f"{dataset_name}: using graph {graph_idx} (rank_sizes={ig.rank_sizes})")

    lifting = CellCycleLifting()
    lifted = lifting.lift_topology(dataset[graph_idx].clone())
    adjacency_1 = lifted["adjacency_1"]
    incidence_2 = lifted["incidence_2"]
    incidence_1_t = lifted["incidence_1"].T

    n0, n1, n2 = ig.rank_sizes[0], ig.rank_sizes[1], ig.rank_sizes[2]
    n_probe = min(n_probe_edges, n1)

    B = ig.B()
    rows = []
    for t in t_values:
        torch.manual_seed(seed)  # same random init at every depth, isolate the effect of t
        x0 = torch.randn(n0, hidden_dim, requires_grad=True)
        x1 = torch.randn(n1, hidden_dim, requires_grad=True)
        x2 = torch.randn(n2, hidden_dim, requires_grad=True)
        model = CWN(hidden_dim, hidden_dim, hidden_dim, hidden_dim, n_layers=t)
        model.eval()

        S_emp = empirical_sensitivity(model, x0, x1, x2, adjacency_1, incidence_2, incidence_1_t, n_probe)
        Bt = bt_matrix(B, t)
        edge_offset = ig.rank_slices[1].start

        for local_sigma in range(n_probe):
            sigma = edge_offset + local_sigma
            for tau in range(n0 + n1 + n2):
                tau_rank = 0 if tau < n0 else (1 if tau < n0 + n1 else 2)
                rows.append({
                    "dataset": dataset_name, "graph_idx": graph_idx, "t": t,
                    "sigma": local_sigma, "tau": tau, "tau_rank": tau_rank,
                    "s_emp": float(S_emp[local_sigma, tau]), "bt_bound": float(Bt[sigma, tau]),
                })

    df = pd.DataFrame(rows)
    sensitivity_dir = osp.join(ANALYSIS_ROOT, "sensitivity")
    os.makedirs(sensitivity_dir, exist_ok=True)
    csv_path = osp.join(sensitivity_dir, f"{dataset_name}_jacobian_vs_bound_graph{graph_idx}.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV written: {csv_path}")

    nonzero_bound = df["bt_bound"] > 0
    frac_nonzero_emp_when_bound_zero = (df.loc[~nonzero_bound, "s_emp"] > 1e-8).mean() if (~nonzero_bound).any() else float("nan")
    print(f"Sparsity check: frac(s_emp > 0 | bt_bound == 0) = {frac_nonzero_emp_when_bound_zero:.4f} (should be ~0: the bound must not miss real gradient flow)")

    for t, sub in df.groupby("t"):
        mask = sub["bt_bound"] > 0
        if mask.sum() >= 3:
            rho, pval = spearmanr(sub.loc[mask, "bt_bound"], sub.loc[mask, "s_emp"])
            print(f"  t={t}: Spearman(bt_bound, s_emp) = {rho:.3f} (p={pval:.3g}, n={mask.sum()})")

    fig, axes = plt.subplots(1, len(t_values), figsize=(4 * len(t_values), 4), squeeze=False)
    for ax, (t, sub) in zip(axes[0], df.groupby("t"), strict=False):
        mask = sub["bt_bound"] > 0
        ax.scatter(sub.loc[mask, "bt_bound"], sub.loc[mask, "s_emp"], alpha=0.5, s=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("(B^t)_{sigma,tau} (structural bound)")
        ax.set_ylabel("||d h_sigma^(t) / d h_tau^(0)||_1 (empirical)")
        ax.set_title(f"t={t}")
    fig.suptitle(f"{dataset_name} graph {graph_idx}: empirical Jacobian vs. structural bound")
    fig.tight_layout()

    figures_dir = osp.join(ANALYSIS_ROOT, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = osp.join(figures_dir, f"{dataset_name}_jacobian_vs_bound.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Figure written: {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MUTAG", choices=["MUTAG", "PROTEINS"])
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--t_values", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_probe_edges", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.dataset, args.hidden_dim, args.t_values, args.n_probe_edges, args.seed)
