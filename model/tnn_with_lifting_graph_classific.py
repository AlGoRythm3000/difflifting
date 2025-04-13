import networkx as nx
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data


import torch.nn.functional as F


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

        self.k_v = data["k_v"]  # Automatically requires_grad=True

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
        data["x_1"]= data["incidence_1"].T @ data["x_0"]
        return data


class TNN_KNN_MLP_G(nn.Module):

    def __init__(self, in_channels, args, hidden_dim, num_classes, k=2, diff_lifting=False, global_pool="sum",
                 device="cpu", tnn_type="SCN2", num_layers_tnn=4, num_layers_gnn=3, embedding_dim=64, k_max=10):
        super(TNN_KNN_MLP_G, self).__init__()
        self.k = k
        self.k_min = 2
        self.k_max = k_max

        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        self.tnn_type = tnn_type
        self.num_classes = num_classes

        #self.feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels], out_channels=hidden_dim,
        #                                             proj_dropout=0.5)
        if diff_lifting:

            if args.gnn == "GIN":
                self.gnn = GIN(in_channels, embedding_dim, embedding_dim, num_layers_gnn).to(device)
            elif args.gnn == "GPS":
                self.gnn = GPS(in_channels, embedding_dim, args.positional_walking_len, num_layers_gnn).to(device)
            self.pool = global_mean_pool
            self.k = k

            if tnn_type in ["UniGCNII", "AST", "HyperGAT", "UniGIN"]:
                self.mlp = nn.Sequential(
                    nn.Linear(embedding_dim, 2 * hidden_dim),
                    nn.ReLU(),
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(hidden_dim, 1),
                )
            if tnn_type not in ["UniGCNII", "AST", "HyperGAT", "UniGIN"]:
                # self.mlp_cell = nn.Sequential(
                #     nn.Linear(k, 2 * hidden_dim),  # Use k as input dimension
                #     nn.ReLU(),
                #     nn.Linear(2 * hidden_dim, hidden_dim),
                #     nn.ReLU(),
                #     nn.Dropout(0.5),
                #     nn.Linear(hidden_dim, k),  # Output k features
                # )
                self.edge_mlp = nn.Sequential(
                    nn.Linear(2 * embedding_dim, 64),  # Input: concatenated edge embeddings
                    nn.ReLU(),
                    nn.Linear(64, 2)  # Output logits for two classes (0 and 1)
                )
                # self.mlp_cell2 = nn.Sequential(
                #     nn.Linear(128, 2 * hidden_dim),  # Change input dimension to 128
                #     nn.ReLU(),
                #     nn.Linear(2 * hidden_dim, hidden_dim),
                #     nn.ReLU(),
                #     nn.Dropout(0.5),
                #     nn.Linear(hidden_dim, 1),  # Output: probability per cycle
                # )
            self.k_mlp = torch.nn.Sequential(
                torch.nn.Linear(embedding_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.k_max - self.k_min + 1)  # Output size covers all possible k values
            )

            self.triangle_count = 0
            self.projection_sum = ProjectionSum()

            #self.attention_lift = AttentionLifting(feature_dim=in_channels, device=device)

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
            #print("Initial data.x shape:", data.x.shape)  # Initial shape
            edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)

            #print("number of edges: ", edge_index_undirected.size(1))
            embeddings = self.gnn(data)

            # Keep gradient through mean operation
            embedding_mean = embeddings

            #print("embeddings requires_grad:", embeddings.requires_grad)

            k_logits = self.k_mlp(embedding_mean)  # Shape: [1, k_max - k_min + 1]


            k_sample = F.gumbel_softmax(k_logits, tau=1.0, hard=True)

            #print("k_sample:", k_sample)

            k_range = torch.arange(self.k_min, self.k_max + 1, device=k_logits.device, dtype=k_logits.dtype)

            #print("k_range:", k_range)


            # Compute the differentiable integer sample as the dot product of the one-hot vector and the range tensor.
            k_v = torch.sum(k_sample * k_range, dim=-1)  # Shape: [1]

            #print("k_v:", k_v)
            #print("k_v grad:", k_v.grad)

            self.k_v = k_v


            mask_knn = torch.nn.functional.one_hot(data.batch_0, num_classes=vertex_slice.shape[0] - 1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings)

            distances = mask_knn * distances + (1 - mask_knn) * 1e7

            if (self.tnn_type == "UniGCNII" or self.tnn_type == "AST" or
                self.tnn_type == "HyperGAT" or self.tnn_type == "UniGIN"):
                knn_indices = torch.topk(-distances, torch.max(self.k_v).long().item(), dim=-1)[1]
                aranged_indices = torch.arange(torch.max(self.k_v).long().item(), device=x.device).expand(self.k_v.shape[0], -1)
                kv_mask = aranged_indices < k_v.unsqueeze(1)
                first_neighbor = knn_indices[:, 0].unsqueeze(1)
                knn_selected = torch.where(kv_mask, knn_indices, first_neighbor)
                knn_indices = knn_selected
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

                node_triangle_matrix = mask.scatter_(1, knn_indices, straight_through_samples.repeat(1, torch.max(self.k_v).long().item()))

                # This is not used, but we keep it for future use
                incidence_matrix_2 = incidence_matrix_1.T @ node_triangle_matrix
                incidence_matrix_2 = torch.div(incidence_matrix_2, 2, rounding_mode='trunc')
                # ---- # ----- # ----- #

                #print("incidence matrix before concatenation:", incidence_matrix_1.shape)
                incidence_matrix_1 = torch.cat((incidence_matrix_1, node_triangle_matrix), dim=1)

                #print("incidence matrix after concatenation:", incidence_matrix_1.shape)
                #print(incidence_matrix_1.grad_fn)
                
                data.x_0 = x.float()

                data.x_0 = torch.div(data.x_0, torch.max(self.k_v))

                #print("data.x_0 after division:", data.x_0.shape)

                data.incidence_1 = incidence_matrix_1
                data.incidence_1 = torch.Tensor(data.incidence_1).to_sparse_coo()

                

                #print(data)
                data = self.feature_encoder(data)
                #print("data after feature encoder", data)
                #print("shapes before tnn: ", data["x_0"].shape)
                tnn_output = self.tnn(data)
                #print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
                out = self.readout(tnn_output, batch)
                #print(out)
                return out["logits"]    

            ## AMAURI E DIEGO, ESSE else é para o celular, olhar o de cima
            else:

                knn_indices = torch.topk(-distances, torch.max(self.k_v).long().item(), dim=-1)[1]
                aranged_indices = torch.arange(torch.max(self.k_v).long().item(), device=x.device).expand(self.k_v.shape[0], -1)
                kv_mask = aranged_indices < k_v.unsqueeze(1)
                first_neighbor = knn_indices[:, 0].unsqueeze(1)
                knn_selected = torch.where(kv_mask, knn_indices, first_neighbor)
                knn_indices = knn_selected
                
                #print("knn_indices: ", knn_indices)

                # Assuming knn_indices is a tensor of shape [num_nodes, k_edges]
                #print("Original knn_indices tensor:")
                #print(knn_indices)  # Display the knn_indices tensor for reference

                # Get the shape: number of nodes and number of candidate neighbors per node
                num_nodes, k_edges = knn_indices.shape
                #print("Step 1: Determine shape of knn_indices")
                #print("Number of nodes:", num_nodes)
                #print("Number of candidate neighbors per node (k_edges):", k_edges)

                # Create a tensor of node indices: [0, 1, 2, ..., num_nodes-1]
                node_indices = torch.arange(num_nodes, device=knn_indices.device)
                #print("Step 2: Create node_indices tensor")
                #print("node_indices before unsqueeze:", node_indices)

                # Unsqueeze to make it a column vector of shape [num_nodes, 1]
                node_indices = node_indices.unsqueeze(1)
                #print("Step 3: node_indices after unsqueeze (shape [num_nodes, 1]):")
                #print(node_indices)

                # Expand node_indices to repeat each node index for each candidate neighbor.
                # After expansion, node_indices has shape [num_nodes, k_edges]
                node_indices = node_indices.expand(num_nodes, k_edges)
                #print("Step 4: node_indices after expand (shape [num_nodes, k_edges]):")
                #print(node_indices)

                # Reshape both node_indices and knn_indices to a flat vector so that each element corresponds to an edge.
                node_indices_flat = node_indices.reshape(-1)
                knn_indices_flat = knn_indices.reshape(-1)
                #print("Step 5: Flatten node_indices and knn_indices:")
                #print("Flattened node_indices:", node_indices_flat)
                #print("Flattened knn_indices:", knn_indices_flat)

                # Stack the flattened node_indices and knn_indices to form an edge_index tensor.
                # The resulting edge_index tensor will have shape [2, num_nodes * k_edges],
                # where the first row is the source node and the second row is the target node.
                edge_indices_knn = torch.stack([node_indices_flat, knn_indices_flat], dim=0)
                #print("Step 6: Final edge_indices_knn tensor (edge index):")
                print(edge_indices_knn)

                source_embeddings = embeddings[edge_indices_knn[0]]  # Shape: [num_selected_edges, embedding_dim]
                target_embeddings = embeddings[edge_indices_knn[1]]  # Shape: [num_selected_edges, embedding_dim]
                edge_embeddings = torch.cat([source_embeddings, target_embeddings], dim=-1)  # Shape: [num_selected_edges, 2 * embedding_dim]

                # Step 2: Compute logits and sharpen probabilities
                edge_logits = self.edge_mlp(edge_embeddings)  # Shape: [num_edges, 2]

                # Define the sharpening factor directly in the code
                sharpening_factor = 10.0  # Example value for sharpening

                # Apply sharpening to logits and compute probabilities
                edge_probs = torch.softmax(edge_logits * sharpening_factor, dim=-1)[:, 1]  # Sharpened probability of class 1

                # Step 3: Apply straight-through estimator
                edge_classes = (edge_probs > 0.5).float()  # Binary values (0 or 1) during forward pass
                edge_classes = edge_classes + (edge_probs - edge_probs.detach())  # Preserve gradients

                # Debugging prints for gradient tracking
                # print("edge_probs requires_grad:", edge_probs.requires_grad)
                # print("edge_probs grad_fn:", edge_probs.grad_fn)
                # print("edge_classes requires_grad:", edge_classes.requires_grad)
                # print("edge_classes grad_fn:", edge_classes.grad_fn)

                # Step 4: Construct the incidence matrix using scatter
                # Step 4: Construct incidence matrix as a sparse tensor with gradients
                num_nodes = embeddings.size(0)
                num_edges_sampled = edge_indices_knn.size(1)
                num_edges= edge_index_undirected.size(1)
                # Prepare indices for sparse incidence matrix
                rows_u = edge_indices_knn[0]  # Source nodes [num_edges_sampled]
                rows_v = edge_indices_knn[1]  # Target nodes [num_edges_sampled]
                cols = torch.arange(num_edges_sampled, device=edge_classes.device)  # Edge indices

                # Stack indices for (u, edge) and (v, edge) pairs
                indices_u = torch.stack([rows_u, cols], dim=0)
                indices_v = torch.stack([rows_v, cols], dim=0)
                all_indices = torch.cat([indices_u, indices_v], dim=1)  # [2, 2*num_edges_sampled]

                # Duplicate edge_classes for both nodes in each edge
                all_values = torch.cat([edge_classes, edge_classes], dim=0)

                # Create sparse incidence matrix with gradient tracking
                incidence_matrix_sampled = torch.sparse_coo_tensor(
                    indices=all_indices,
                    values=all_values,
                    size=(num_nodes, num_edges_sampled),
                    device=edge_classes.device
                ).coalesce()

                # Now combine with original edges
                original_incidence = torch.zeros(
                    (num_nodes, num_edges), 
                    device=embeddings.device
                )
                for idx, edge in enumerate(edge_index_undirected.T):
                    original_incidence[edge[0], idx] = 1
                    original_incidence[edge[1], idx] = 1

                # Convert to sparse and concatenate
                original_incidence_sparse = original_incidence.to_sparse_coo()
                incidence_matrix_1 = torch.cat(
                    [original_incidence_sparse, incidence_matrix_sampled], 
                    dim=1
                )

                # Ensure gradients flow through values
                incidence_matrix_1 = incidence_matrix_1.coalesce()
                incidence_matrix_1.requires_grad_(True)  # Explicitly enable gradients

                # Pass through TNN
                data_for_lifting = {
                    "x_0": embeddings,  # Use GNN embeddings instead of raw features
                    "incidence_1": incidence_matrix_1,
                    "adjacency_1": (incidence_matrix_1 @ incidence_matrix_1.T).coalesce(),
                }
                
                # print("incidence_matrix_1 requires_grad:", incidence_matrix_1.requires_grad)
                # print("incidence_matrix_sampled requires_grad:", incidence_matrix_sampled.requires_grad)
                # print("Gradient function of incidence_matrix_1:", incidence_matrix_1.grad_fn)
                # print("Gradient function of incidence_matrix_sampled:", incidence_matrix_sampled.grad_fn)
                # print("incidence_matrix_1:", incidence_matrix_1)
                # print("incidence_matrix_sampled:", incidence_matrix_sampled)
                # # Step 1: compute edge-edge adjacency via shared face
                # Step 1: Build adjacency matrix
                # Step 1: Build adjacency matrix
                A = incidence_matrix_1.T @ incidence_matrix_1  # [num_edges, num_edges]

                # Step 2: Remove diagonal (self-loops) using element-wise multiplication
                num_tot_edges = A.size(0)
                identity_indices = torch.arange(num_tot_edges, device=A.device).repeat(2, 1)
                identity_values = torch.ones(num_tot_edges, device=A.device)
                identity_mask = torch.sparse_coo_tensor(identity_indices, identity_values, size=A.size()).coalesce()

                # Perform element-wise multiplication to remove diagonal entries
                A = A * (1 - identity_mask.to_dense())

                # Step 3: Replace all 2s with 1, keeping 0s and 1s untouched (differentiably!)
                A = torch.sparse_coo_tensor(
                    A.indices(),
                    torch.clamp(A.values(), max=1),
                    A.size(),
                    device=A.device
                ).coalesce()
                # Step 4: Store
                data.adjacency_1 = A

                # Optional: print check
                #print("Unique values in adjacency_1:", A.unique())

                #print(data.adjacency_1)
                #print(data.adjacency_1.shape)
                # incidence_matrix_1 = torch.Tensor(incidence_matrix_1).to_sparse_coo()

                #embeddings[]
                # node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,3))

                # incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix
                # incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')

                data_for_lifting = {}

                data_for_lifting = {
                    "x_0": x.float(),  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                    "incidence_2": incidence_matrix_1.T,  # edge_to-triangle
                    "adjacency_1": A
                }

                lifted_data = self.projection_sum(data_for_lifting)
                #print("Lifted data keys:", lifted_data.keys())
                #print(lifted_data)
                # for key, value in lifted_data.items():
                #     print(f"Key: {key}, Shape: {value.shape if isinstance(value, torch.Tensor) else 'Not a Tensor'}")
                # print("Data keys:", data.keys)
                # print(data)

                #data = self.__create_laplacians(data, incidence_matrix_1, lifted_data, data_for_lifting)
                
                lifted_data["adjacency_1"] = A
                #print(lifted_data)
                lifted_data["x_0"] = torch.div(lifted_data["x_0"], torch.max(self.k_v))
                #print(lifted_data)
                lifted_data_obj = Data(**lifted_data)
                tnn_output = self.tnn(lifted_data_obj)
                # print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
                batch["incidence_1"]= incidence_matrix_1

                batch["incidence_2"]= incidence_matrix_1.T
                out = self.readout(tnn_output, batch)
                # print(out)
                return out["logits"]

        
        #print("data ": data)
        #data = self.feature_encoder(data)
        # print("data after feature encoder", data)
        # print("shapes before tnn: ", data["x_0"].shape)
        tnn_output = self.tnn(data)
        # print("shapes tnn: ", tnn_output["x_0"].shape, tnn_output["x_1"].shape)
        out = self.readout(tnn_output, batch)
        # print(out)
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
