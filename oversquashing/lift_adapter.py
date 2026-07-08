"""Bridge between the static liftings in `tools/lifting/` and the influence graph.

Each static lifting (cellular/simplicial via TopoNetX, or hand-assembled
hypergraph) returns a dict with `incidence_k` keys (k >= 1), where
`incidence_k` = B_k : (k-1)-cells -> k-cells. That's the only thing
`influence_graph.build_influence_graph` needs.
"""

import numpy as np
import scipy.sparse as sp
import torch_geometric

from oversquashing.influence_graph import InfluenceGraph, build_influence_graph


def get_incidences_from_lifted_data(lifted_data: dict) -> list:
    """Extracts [B_1, B_2, ...] (in rank order) from a `lift_topology()` dict.

    Ignores `incidence_0` (always an empty (0, n_nodes) placeholder on the
    TopoNetX side, not a real boundary relation) and any composite key of
    the form `incidence_up-1` produced by `select_neighborhoods_of_interest`
    (not used by this repo's static liftings, but guarded against anyway).
    """
    ranks = sorted(
        int(key.split("_")[1])
        for key in lifted_data
        if key.startswith("incidence_") and key.split("_")[1].isdigit()
    )
    ranks = [r for r in ranks if r >= 1]
    if not ranks:
        raise ValueError(
            "No 'incidence_k' key (k>=1) found in the lifted topology."
        )
    return [lifted_data[f"incidence_{r}"] for r in ranks]


def build_influence_graph_from_lifting(
    lifting, data: torch_geometric.data.Data
) -> InfluenceGraph:
    """Applies a static lifting to a graph then builds its influence graph.

    Parameters
    ----------
    lifting : tools.lifting.abstract_lifting.AbstractLifting
        Instance of a static lifting (e.g. CellCycleLifting(), HypergraphKHopLifting()).
    data : torch_geometric.data.Data
        The input graph (a single, unbatched graph).

    Returns
    -------
    InfluenceGraph
    """
    lifted_topology = lifting.lift_topology(data)
    incidences = get_incidences_from_lifted_data(lifted_topology)
    return build_influence_graph(incidences)


def build_influence_graph_levels(
    lifting, data: torch_geometric.data.Data
) -> dict:
    """3-level ablation to isolate 1-skeleton rewiring from the addition of
    higher-order cells.

    - "base"    : (a) original graph, un-lifted edges (`build_influence_graph_raw`).
    - "rewired" : (b) lifted 1-skeleton alone (the lifting's `incidence_1`,
      dropping `incidence_2+`) -- zero rank >= 2 cells.
    - "complex" : (c) full lifted complex (every rank the lifting produces).

    This is a **generic truncation**, valid for any lifting with no
    domain-specific code: for liftings that preserve the 1-skeleton
    (`CellCycleLifting`, clique), "rewired" == "base" by construction (see
    the consistency test in `scripts/ablation_levels_sanity.py`). For
    hypergraph liftings (ranks 0/1 only), "rewired" == "complex" trivially
    (there's no `incidence_2` to drop).
    """
    lifted_topology = lifting.lift_topology(data.clone())
    incidences = get_incidences_from_lifted_data(lifted_topology)
    return {
        "base": build_influence_graph_raw(data),
        "rewired": build_influence_graph(incidences[:1]),
        "complex": build_influence_graph(incidences),
    }


def raw_graph_incidence(data: torch_geometric.data.Data) -> list:
    """Builds B_1 (nodes x edges) directly from `edge_index`, with no
    lifting at all: this is the "raw GNN" / message-passing-on-the-1-skeleton
    reference (ranks 0/1, no higher-order cell), used as a control baseline.
    """
    edge_index = data.edge_index.detach().cpu().numpy()
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(edge_index.max()) + 1
    pairs = sorted({
        (int(u), int(v)) if u < v else (int(v), int(u))
        for u, v in zip(edge_index[0], edge_index[1])
        if u != v
    })
    rows, cols = [], []
    for j, (u, v) in enumerate(pairs):
        rows += [u, v]
        cols += [j, j]
    values = np.ones(len(rows))
    B_1 = sp.csr_matrix((values, (rows, cols)), shape=(num_nodes, len(pairs)))
    return [B_1]


def build_influence_graph_raw(data: torch_geometric.data.Data) -> InfluenceGraph:
    """Influence graph of the 1-skeleton alone ("raw GNN" reference, no lifting)."""
    return build_influence_graph(raw_graph_incidence(data))
