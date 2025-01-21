import networkx as nx
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_undirected

from layers.encoders.all_cell_features_encoders import AllCellFeatureEncoder
from model.GNN import GIN, GPS
# from layers.diff_lifting import DiffLifting
from model.TNN import TNN
from model.tnn_with_lifiting import ProjectionSum
from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import DirectReadout
from tools.redout import PropagateSignalDown


class AttentionLifting(nn.Module):
    """Lift node features to hyperedge features using attention mechanism."""

    def __init__(self, feature_dim=None, device="cpu"):
        super().__init__()
        self.W1 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.W2 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.W3 = torch.randn(feature_dim, feature_dim, device=device) if feature_dim else torch.randn(64, 64,
                                                                                                       device=device)
        self.k_v = torch.tensor(2.0)  # Automatically requires_grad=True
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(feature_dim if feature_dim else 64, feature_dim * 2 if feature_dim else 128),
            torch.nn.ReLU(),
            torch.nn.Linear(feature_dim * 2 if feature_dim else 128, feature_dim if feature_dim else 64)
        ).to(device)

    def print_grad_W1(grad):
        print("Gradient for W1:", grad)

    def print_grad_W2(grad):
        print("Gradient for W2:", grad)

    def print_grad_W3(grad):
        print("Gradient for W3:", grad)

    def print_grad_kv(grad):
        print("Gradient for k_v:", grad)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, data):
        """Project node features to higher-order structures using attention."""
        keys = sorted(
            [key.split("_")[1] for key in data if ("incidence" in key and "-" not in key)]
        )

        self.k_v = torch.nn.Parameter(torch.tensor(2.0))  # Automatically requires_grad=True

        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                incidence = data["incidence_" + elem]

                # Get nodes involved in each structure
                node_features = data[f"x_{idx_to_project}"]

                # For each structure (edge/hyperedge), get its incident nodes
                structures = []
                for i in range(incidence.shape[1]):
                    nodes = torch.where(incidence[:, i] != 0)[0]
                    if len(nodes) > 0:
                        structures.append((i, nodes))

                # Compute lifted features for each structure
                lifted_features = []
                for struct_idx, nodes in structures:
                    features = node_features[nodes]

                    # Compute attention scores using the gradient-preserving k_v
                    scaling_factor = torch.sqrt(self.k_v)
                    query = torch.matmul(features, self.W1.t())
                    key = torch.matmul(features, self.W2.t())
                    scores = torch.matmul(query, key.t()) / scaling_factor

                    # Apply attention
                    attention = torch.softmax(scores, dim=-1)
                    values = torch.matmul(features, self.W3.t())
                    messages = torch.matmul(attention, values)

                    # Apply order-invariant aggregation
                    structure_feature = self.phi(messages.mean(dim=0, keepdim=True))
                    lifted_features.append(structure_feature)

                # Combine all lifted features
                if lifted_features:
                    data["x_" + elem] = torch.cat(lifted_features, dim=0)
                else:
                    data["x_" + elem] = torch.zeros(
                        (incidence.shape[1], node_features.shape[1]),
                        device=node_features.device
                    )

        return data

    def forward(self, data):
        """Apply the lifting to the input data."""
        data = self.lift_features(data)
        return data


class TNN_KNN_MLP_G(nn.Module):

    def __init__(self, in_channels, args, hidden_dim, num_classes, k=2, diff_lifting=False, global_pool="sum",
                 device="cpu", tnn_type="SCN2", num_layers_tnn=4, num_layers_gnn=3, embedding_dim=64):
        super(TNN_KNN_MLP_G, self).__init__()
        self.k = k
        self.k_min = 2
        self.k_max = 10

        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        self.tnn_type = tnn_type
        self.num_classes = num_classes

        self.feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels], out_channels=hidden_dim,
                                                     proj_dropout=0.5)
        if diff_lifting:

            if args.gnn == "GIN":
                self.gnn = GIN(in_channels, embedding_dim, embedding_dim, num_layers_gnn).to(device)
            elif args.gnn == "GPS":
                self.gnn = GPS(in_channels, embedding_dim, args.positional_walking_len, num_layers_gnn).to(device)
            self.pool = global_mean_pool
            self.k = k
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, 2 * hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_dim, 1),
            )
            if tnn_type not in ["UniGCNII", "AST"]:
                self.mlp_cell = nn.Sequential(
                    nn.Linear(k, 2 * hidden_dim),  # Use k as input dimension
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, k),  # Output k features
                )
                self.mlp_cell2 = nn.Sequential(
                    nn.Linear(128, 2 * hidden_dim),  # Change input dimension to 128
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, 1),  # Output: probability per cycle
                )
            self.k_mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.k_max - self.k_min + 1)  # Output size covers all possible k values
            )

            self.triangle_count = 0
            self.projection_sum = ProjectionSum()

            self.attention_lift = AttentionLifting(feature_dim=in_channels, device=device)

        self.tnn = TNN(
            model_type=tnn_type,  # choose TNN model
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            n_layers=num_layers_tnn,
            device=device
        )
        if args.no_readout:
            self.readout = DirectReadout(**{
                "readout_name": "DirectReadout",
                "task_level": "graph",
                "hidden_dim": hidden_dim,
                "out_channels": num_classes,
            })
        else:
            self.readout = PropagateSignalDown(**{
                "readout_name": "PropagateSignalDownLinear",
                "num_cell_dimensions": 3,
                "hidden_dim": hidden_dim,
                "out_channels": num_classes,
                "task_level": "graph",
                "pooling_type": global_pool,
            })

    def __create_laplacians(self, data, incidence_matrix_1, lifted_data, data_for_lifting):
        new_edge_index, new_edge_attr = torch_geometric.utils.get_laplacian(data.edge_index)
        data.x_1 = lifted_data["x_1"]
        data.x_2 = lifted_data["x_2"]
        laplacian_0 = torch.sparse_coo_tensor(
            indices=new_edge_index,
            values=new_edge_attr,
            size=(data.x.shape[0], data.x.shape[0])
        )

        data.laplacian_up_0 = laplacian_0
        data.laplacian_up_1 = torch.spmm(data_for_lifting["incidence_2"],
                                         data_for_lifting["incidence_2"].T).to_sparse_coo()
        data.laplacian_down_1 = torch.spmm(data_for_lifting["incidence_1"].T,
                                           data_for_lifting["incidence_1"]).to_sparse_coo()

        data.laplacian_down_2 = torch.spmm(data_for_lifting["incidence_2"].T,
                                           data_for_lifting["incidence_2"]).to_sparse_coo()
        data.node_edge_matrix = incidence_matrix_1

        data.incidence_1 = data_for_lifting.get("incidence_1")
        data.incidence_2 = data_for_lifting.get("incidence_2")

        data.hodge_laplacian_0 = data.laplacian_up_0  # + data.laplacian_down_0
        data.hodge_laplacian_1 = data.laplacian_up_1 + data.laplacian_down_1
        data.hodge_laplacian_2 = data.laplacian_down_2
        return data

    def forward(self, batch):
        data = batch
        if self.diff_lifting:
            x, edge_index = data.x.float(), data.edge_index
            edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)
            embeddings = self.gnn(data)

            # Keep gradient through mean operation
            embedding_mean = embeddings.mean(dim=0, keepdim=True)

            print("embeddings requires_grad:", embeddings.requires_grad)

            k_logits = self.k_mlp(embedding_mean)  # Shape: [1, k_max - k_min + 1]
            k_logits_sum = k_logits.sum()

            mask_knn = torch.nn.functional.one_hot(data.batch_0, num_classes=vertex_slice.shape[0] - 1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings)

            distances = mask_knn * distances + (1 - mask_knn) * 1e7

            if self.tnn_type == "UniGCNII" or self.tnn_type == "AST":
                knn_indices = torch.topk(-distances, self.k, dim=-1)[1]

                pooled_embeddings = embeddings[knn_indices.long()].mean(axis=1, keepdim=True).squeeze()

                include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
                inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
                straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())

                num_edges = edge_index_undirected.size(1)

                incidence_matrix_1 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)

                for idx, edge in enumerate(edge_index_undirected.T):
                    incidence_matrix_1[edge[0], idx] = 1
                    incidence_matrix_1[edge[1], idx] = 1

                num_nodes = data.x.size(0)

                mask = torch.zeros((num_nodes, num_nodes), device=data.x.device)

                node_triangle_matrix = mask.scatter_(1, knn_indices, straight_through_samples.repeat(1, self.k))

                # This is not used, but we keep it for future use
                incidence_matrix_2 = incidence_matrix_1.T @ node_triangle_matrix
                incidence_matrix_2 = torch.div(incidence_matrix_2, 2, rounding_mode='trunc')
                # ---- # ----- # ----- #

                data_for_lifting = {}

                incidence_matrix_1 = torch.cat((incidence_matrix_1, node_triangle_matrix), dim=1)
                print(incidence_matrix_1.grad_fn)
                data_for_lifting = {
                    "x_0": x.float(),
                    "incidence_1": incidence_matrix_1,
                    "k_v": k_logits_sum  # Pass the continuous k value
                }

                lifted_data = self.attention_lift(data_for_lifting)

                data.x_0 = x.float()

                data.incidence_1 = data_for_lifting.get("incidence_1")
                data.incidence_1 = torch.Tensor(data.incidence_1).to_sparse_coo()

            ## AMAURI E DIEGO, ESSE else é para o celular, olhar o de cima
            else:
                knn_indices = torch.topk(-distances, self.k, dim=-1)[1]

                print("knn_indices: ", knn_indices.shape, knn_indices)

                num_nodes, k = knn_indices.size()
                embedding_dim = embeddings.size(1)

                knn_embeddings = embeddings[knn_indices]  # Shape: [num_nodes, k, embedding_dim]
                print("knn_embeddings: ", knn_embeddings.shape, knn_embeddings)
                central_embeddings = embeddings.unsqueeze(1).expand(-1, k, -1)  # Shape: [num_nodes, k, embedding_dim]
                print("central_embeddings: ", central_embeddings.shape, central_embeddings)
                edge_embeddings = torch.cat([central_embeddings, knn_embeddings],
                                            dim=-1)  # Shape: [num_nodes, k, 2 * embedding_dim]
                print("edge_embeddings: ", edge_embeddings.shape, edge_embeddings)
                pooled_embeddings = edge_embeddings.mean(dim=2)  # Shape: [num_nodes, k, embedding_dim]
                print("pooled_embeddings: ", pooled_embeddings.shape, pooled_embeddings)
                include_probs = torch.sigmoid(self.mlp_cell(pooled_embeddings))  # Shape: [num_nodes, k]
                print("include_probs: ", include_probs.shape, include_probs)
                inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()  # Shape: [num_nodes, k]
                print("inclusion_samples: ", inclusion_samples.shape, inclusion_samples)
                straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())
                print("straight_through_samples: ", straight_through_samples.shape, straight_through_samples)
                num_new_edges = num_nodes * k
                new_sampled_incidence = torch.zeros((num_nodes, num_new_edges), device=embeddings.device)
                print("new_sampled_incidence: ", new_sampled_incidence.shape, new_sampled_incidence)
                edge_indices = torch.arange(num_new_edges, device=embeddings.device).view(num_nodes,
                                                                                          k)  # Shape: [num_nodes, k]
                print("edge_indices: ", edge_indices.shape, edge_indices)
                print("edge_indices view", edge_indices.view(-1, 1).shape)
                print("straight view", straight_through_samples.view(-1, 1).shape)
                new_sampled_incidence.scatter_(1, edge_indices.view(-1, 1).T, straight_through_samples.view(-1, 1).T)
                print("new_sampled_incidence: ", new_sampled_incidence.shape, new_sampled_incidence)

                num_edges = edge_index_undirected.size(1)

                original_incidence_1 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)

                for idx, edge in enumerate(edge_index_undirected.T):
                    original_incidence_1[edge[0], idx] = 1
                    original_incidence_1[edge[1], idx] = 1

                final_incidence = torch.cat([original_incidence_1, new_sampled_incidence],
                                            dim=1)  # Shape: [num_nodes, num_original_edges + num_new_edges]

                print("final_incidence: ", final_incidence.shape, final_incidence)

                # (Optionally) Clone final_incidence if needed for future gradient tracking
                final_incidence_1 = final_incidence.clone()  # For potential future gradient use
                print("Cloned final_incidence_1: ", final_incidence_1.shape, final_incidence_1)

                k = edge_indices.size(1)  # Should be 3

                edge_indices = torch.nonzero(final_incidence, as_tuple=True)  # Returns indices of non-zero entries
                rows, cols = edge_indices  # Rows are node indices, cols are edge indices
                print("original edge_index undirected: ", edge_index_undirected.shape)
                # Create a dictionary to map each edge index to its corresponding pair of nodes
                edge_dict = {}
                for node, edge in zip(rows.tolist(), cols.tolist()):
                    if edge not in edge_dict:
                        edge_dict[edge] = [node]
                    else:
                        edge_dict[edge].append(node)

                edges = [tuple(nodes) for nodes in edge_dict.values() if
                         len(nodes) == 2]  # Ensure only valid edges are considered

                G = nx.Graph()
                G.add_edges_from(edges)

                print(f"Number of nodes: {G.number_of_nodes()}")
                print(f"Number of edges: {G.number_of_edges()}")
                print(f"Cycles: {nx.cycle_basis(G)}")

                edge_indices = torch.nonzero(final_incidence, as_tuple=True)  # Indices of non-zero entries
                rows, cols = edge_indices  # Rows are node indices, cols are edge indices

                edge_to_nodes = {}
                for edge_idx in cols.unique():
                    nodes = rows[cols == edge_idx].tolist()
                    if len(nodes) == 2:  # Only consider valid edges with exactly two endpoints
                        edge_to_nodes[edge_idx.item()] = tuple(nodes)

                edges_tensor = torch.tensor(list(edge_to_nodes.values()),
                                            device=final_incidence.device)  # Shape: [num_edges, 2]

                subgraphs = []
                for node in range(num_nodes):
                    neighbors = set(knn_indices[node].tolist())
                    neighbors.add(node)
                    neighbors_tensor = torch.tensor(list(neighbors), device=final_incidence.device)

                    mask = (torch.isin(edges_tensor[:, 0], neighbors_tensor) &
                            torch.isin(edges_tensor[:, 1], neighbors_tensor))
                    subgraph_edges = edges_tensor[mask]  # Shape: [num_filtered_edges, 2]

                    G = nx.Graph()
                    G.add_edges_from(subgraph_edges.tolist())
                    subgraphs.append(G)

                print(f"Number of subgraphs: {len(subgraphs)}")
                for i, G in enumerate(subgraphs):
                    print(f"Subgraph {i}: {G.edges()}")  # Print edges of each subgraph
                    print(f"Cycles in Subgraph {i}: {nx.cycle_basis(G)}")

                subgraph_id = 1913
                selected_graph = subgraphs[subgraph_id]
                print(f"Subgraph {subgraph_id} edges: {selected_graph.edges()}")

                print(nx.cycle_basis(selected_graph))

                print(f"Number of nodes in Subgraph {subgraph_id}: {selected_graph.number_of_nodes()}")
                print(f"Number of edges in Subgraph {subgraph_id}: {selected_graph.number_of_edges()}")
                print(f"Cycles in Subgraph {subgraph_id}: {nx.cycle_basis(selected_graph)}")
                # cell_cycle_lifting = CellCycleLifting(max_cell_length=6)  # Define max cell length as needed
                # lifted_topology = cell_cycle_lifting.lift_topology(data_geometric)

                # print("Lifted Topology: ", lifted_topology.keys())
                # print("Lifted Topology incidence: ", lifted_topology["incidence_1"])
                # print("Lifted Topology incidence: ", lifted_topology["incidence_2"])

                # Step 7: Cycle Sampling for Cell Construction

                # pooled_embeddings = embeddings[knn_indices.long()].mean(axis=1, keepdim=True).squeeze()

                # include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
                # inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
                # straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())

                # mask = torch.zeros((num_nodes, num_nodes),device=data.x.device)

                # node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,3))

                # incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix
                # incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')

                data_for_lifting = {}

                data_for_lifting = {
                    "x_0": x.float(),  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                    "incidence_2": incidence_matrix_2,  # edge_to-triangle
                }

                lifted_data = self.projection_sum(data_for_lifting)

                data.x_0 = x.float()

                data = self.__create_laplacians(data, incidence_matrix_1, lifted_data, data_for_lifting)

        data = self.feature_encoder(data)
        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)

        return out["logits"]


def generate_graph_from_data(
        data: torch_geometric.data.Data
) -> nx.Graph:
    r"""Generate a NetworkX graph from the input data object.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data.

    Returns
    -------
    nx.Graph
        The generated NetworkX graph.
    """
    # Check if data object have edge_attr, return list of tuples as [(node_id, {'features':data}, 'dim':1)] or ??
    nodes = [
        (n, dict(features=data.x[n], dim=0))
        for n in range(data.x.shape[0])
    ]

    if hasattr(data, "edge_attr"):
        # In case edge features are given, assign features to every edge
        edge_index, edge_attr = (
            data.edge_index,
            (
                data.edge_attr
                if is_undirected(data.edge_index, data.edge_attr)
                else to_undirected(data.edge_index, data.edge_attr)
            ),
        )
        edges = [
            (i.item(), j.item(), dict(features=edge_attr[edge_idx], dim=1))
            for edge_idx, (i, j) in enumerate(
                zip(edge_index[0], edge_index[1], strict=False)
            )
        ]
    else:
        # If edge_attr is not present, return list list of edges
        edges = [
            (i.item(), j.item(), {})
            for i, j in zip(
                data.edge_index[0], data.edge_index[1], strict=False
            )
        ]
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
