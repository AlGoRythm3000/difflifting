import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, DeepSetsAggregation
import torch.nn.functional as F
from torch_geometric.utils import degree
import networkx as nx
from tools.lifting.cycle_lifting import CellCycleLifting
from torch_geometric.utils import is_undirected
from torch_geometric.utils import to_undirected
from torch_geometric.utils import to_networkx


import torch_geometric

from layers.deepset import DeepSetLayer
from layers.encoders.all_cell_features_encoders import AllCellFeatureEncoder
from model.GNN import GIN, GPS
from model.TNN import TNN
from torch_geometric.transforms import BaseTransform

from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import PropagateSignalDown


from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

# from layers.diff_lifting import DiffLifting
from model.TNN import TNN
from torch_geometric.transforms import BaseTransform

from model.tnn_with_lifiting import ProjectionSum
from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import PropagateSignalDown
from tools.redout import DirectReadout




class TNN_KNN_MLP_G(nn.Module):

    def __init__(self,in_channels, args, hidden_dim, num_classes, k=2, diff_lifting=False,global_pool="sum",
                 device="cpu", tnn_type= "SCN2", num_layers_tnn=4, num_layers_gnn=3, embedding_dim=64):
        super(TNN_KNN_MLP_G, self).__init__()
        self.k = k
        self.k_min = 2
        self.k_max = 10

        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        self.tnn_type = tnn_type
        self.num_classes = num_classes
        #self.lin= nn.Linear(tnn_out_feat, num_classes)
        self.feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels, in_channels, in_channels], out_channels=hidden_dim, proj_dropout=0.5)
        if diff_lifting:
            # self.deep_set_layers = []
            # for i in range(k):
            #     self.deep_set_layers.append(DeepSetLayer(in_channels, hidden_dim))
            if args.gnn == "GIN":
                self.gnn = GIN(in_channels, embedding_dim, embedding_dim, num_layers_gnn).to(device)
            elif args.gnn == "GPS":
                self.gnn = GPS(in_channels, embedding_dim, args.positional_walking_len , num_layers_gnn).to(device)
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
            self.mlp_cell  = nn.Sequential(
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

    def __create_laplacians(self, data,incidence_matrix_1, lifted_data, data_for_lifting):
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

            # Generate k distribution from embeddings
            k_logits = self.k_mlp(embeddings.mean(dim=0, keepdim=True))  # Shape: [1, k_max - k_min + 1]
            
            # Use PyTorch's gumbel_softmax for differentiable sampling
            k_sample = torch.nn.functional.gumbel_softmax(k_logits, tau=1.0, hard=True, dim=-1)
            
            # Convert one-hot to scalar k value
            k_values = torch.arange(self.k_min, self.k_max + 1, device=k_logits.device)
            k = (k_sample * k_values).sum()  # Weighted sum gives us our k value
            
            # Store k for topk operation
            self.k = max(2, int(k.item()))

            mask_knn= torch.nn.functional.one_hot(data.batch_0,num_classes=vertex_slice.shape[0]-1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings) # Find if there is cdist without sqrt

            distances= mask_knn * distances + (1-mask_knn)*1e7

            if self.tnn_type == "UniGCNII" or self.tnn_type=="AST":
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

                num_nodes= data.x.size(0)

                print("OUUUUUUU\n")


                mask = torch.zeros((num_nodes, num_nodes),device=data.x.device)

                node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,self.k))

                incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix
                incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')

                data_for_lifting={}

                incidence_matrix_1 = torch.cat((incidence_matrix_1, node_triangle_matrix), dim=1)
                data_for_lifting = {
                    "x_0": x.float(),  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                }
                print("OUUUUUUU\n")
                lifted_data = self.projection_sum(data_for_lifting)

                data.x_0 = x.float()

                data.incidence_1 = data_for_lifting.get("incidence_1")
                data.incidence_1= torch.Tensor(data.incidence_1).to_sparse_coo()
            

            else:
                knn_indices = torch.topk(-distances, self.k, dim=-1)[1]

                print("knn_indices: ", knn_indices.shape, knn_indices)

                                # Assume knn_indices, embeddings, and original_incidence are given
                num_nodes, k = knn_indices.size()
                embedding_dim = embeddings.size(1)

                # Step 1: Gather embeddings for each node's KNN
                knn_embeddings = embeddings[knn_indices]  # Shape: [num_nodes, k, embedding_dim]
                print("knn_embeddings: ", knn_embeddings.shape, knn_embeddings)
                # Step 2: Concatenate the central node embedding with its KNN embeddings
                # For each node v0, create pairs: [embed(v0), embed(v1)], ..., [embed(v0), embed(vk)]
                central_embeddings = embeddings.unsqueeze(1).expand(-1, k, -1)  # Shape: [num_nodes, k, embedding_dim]
                print("central_embeddings: ", central_embeddings.shape, central_embeddings)
                edge_embeddings = torch.cat([central_embeddings, knn_embeddings], dim=-1)  # Shape: [num_nodes, k, 2 * embedding_dim]
                print("edge_embeddings: ", edge_embeddings.shape, edge_embeddings)
                # Step 3: Pool the embeddings for each new edge (e.g., mean or sum pooling)
                pooled_embeddings = edge_embeddings.mean(dim=2)  # Shape: [num_nodes, k, embedding_dim]
                print("pooled_embeddings: ", pooled_embeddings.shape, pooled_embeddings)
                # Step 4: Use an MLP to compute inclusion probabilities
                include_probs = torch.sigmoid(self.mlp_cell(pooled_embeddings))  # Shape: [num_nodes, k]
                print("include_probs: ", include_probs.shape, include_probs)
                # Step 5: Apply the straight-through estimator to sample inclusion
                inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()  # Shape: [num_nodes, k]
                print("inclusion_samples: ", inclusion_samples.shape, inclusion_samples)
                straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())
                print("straight_through_samples: ", straight_through_samples.shape, straight_through_samples)
                # Step 6: Create the new incidence matrix
                # Initialize the new_sampled_incidence as a zero matrix
                num_new_edges = num_nodes * k
                new_sampled_incidence = torch.zeros((num_nodes, num_new_edges), device=embeddings.device)
                print("new_sampled_incidence: ", new_sampled_incidence.shape, new_sampled_incidence)
                # Map node indices to edge indices
                edge_indices = torch.arange(num_new_edges, device=embeddings.device).view(num_nodes, k)  # Shape: [num_nodes, k]
                print("edge_indices: ", edge_indices.shape, edge_indices)
                # Scatter the straight-through samples into the incidence matrix
                print("edge_indices view", edge_indices.view(-1, 1).shape)
                print("straight view", straight_through_samples.view(-1, 1).shape)
                new_sampled_incidence.scatter_(1, edge_indices.view(-1, 1).T, straight_through_samples.view(-1, 1).T)
                print("new_sampled_incidence: ", new_sampled_incidence.shape, new_sampled_incidence)
                # Step 7: Concatenate the original and new incidence matrices

                num_edges = edge_index_undirected.size(1)

                original_incidence_1 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)
                
                for idx, edge in enumerate(edge_index_undirected.T):
                    original_incidence_1[edge[0], idx] = 1
                    original_incidence_1[edge[1], idx] = 1

                final_incidence = torch.cat([original_incidence_1, new_sampled_incidence], dim=1)  # Shape: [num_nodes, num_original_edges + num_new_edges]

                print("final_incidence: ", final_incidence.shape, final_incidence)

                # (Optionally) Clone final_incidence if needed for future gradient tracking
                final_incidence_1 = final_incidence.clone()  # For potential future gradient use
                print("Cloned final_incidence_1: ", final_incidence_1.shape, final_incidence_1)

                k = edge_indices.size(1)  # Should be 3

                # Step 1: Get the edges from the incidence matrix
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

                # Convert the edge_dict to a list of edges
                edges = [tuple(nodes) for nodes in edge_dict.values() if len(nodes) == 2]  # Ensure only valid edges are considered

                # Step 2: Create the NetworkX graph
                G = nx.Graph()
                G.add_edges_from(edges)
    
                # Step 3: Print or analyze the graph
                print(f"Number of nodes: {G.number_of_nodes()}")
                print(f"Number of edges: {G.number_of_edges()}")
                print(f"Cycles: {nx.cycle_basis(G)}")

                # Precompute edge-endpoint mappings
                edge_indices = torch.nonzero(final_incidence, as_tuple=True)  # Indices of non-zero entries
                rows, cols = edge_indices  # Rows are node indices, cols are edge indices

                # Convert the incidence matrix into an edge-to-node mapping
                edge_to_nodes = {}
                for edge_idx in cols.unique():
                    nodes = rows[cols == edge_idx].tolist()
                    if len(nodes) == 2:  # Only consider valid edges with exactly two endpoints
                        edge_to_nodes[edge_idx.item()] = tuple(nodes)

                # Convert edge_to_nodes into a tensor for efficient processing
                edges_tensor = torch.tensor(list(edge_to_nodes.values()), device=final_incidence.device)  # Shape: [num_edges, 2]

                # Create subgraphs as NetworkX undirected graphs
                subgraphs = []
                for node in range(num_nodes):
                    # Get the neighbors (including the node itself)
                    neighbors = set(knn_indices[node].tolist())
                    neighbors.add(node)
                    
                    # Convert neighbors to a tensor for efficient comparison
                    neighbors_tensor = torch.tensor(list(neighbors), device=final_incidence.device)
                    
                    # Filter edges where both endpoints are in the neighbor set
                    mask = (torch.isin(edges_tensor[:, 0], neighbors_tensor) &
                            torch.isin(edges_tensor[:, 1], neighbors_tensor))
                    subgraph_edges = edges_tensor[mask]  # Shape: [num_filtered_edges, 2]
                    
                    # Convert to a NetworkX undirected graph
                    G = nx.Graph()
                    G.add_edges_from(subgraph_edges.tolist())
                    
                    # Add the graph to the list of subgraphs
                    subgraphs.append(G)

                # Analyze the subgraphs
                print(f"Number of subgraphs: {len(subgraphs)}")
                for i, G in enumerate(subgraphs):
                    print(f"Subgraph {i}: {G.edges()}")  # Print edges of each subgraph
                    print(f"Cycles in Subgraph {i}: {nx.cycle_basis(G)}")


                # Example analysis for a specific subgraph
                subgraph_id = 1913  # Change to the index of the desired subgraph
                selected_graph = subgraphs[subgraph_id]
                print(f"Subgraph {subgraph_id} edges: {selected_graph.edges()}")

                print(nx.cycle_basis(selected_graph))

                # Additional analysis
                print(f"Number of nodes in Subgraph {subgraph_id}: {selected_graph.number_of_nodes()}")
                print(f"Number of edges in Subgraph {subgraph_id}: {selected_graph.number_of_edges()}")
                print(f"Cycles in Subgraph {subgraph_id}: {nx.cycle_basis(selected_graph)}")
                #cell_cycle_lifting = CellCycleLifting(max_cell_length=6)  # Define max cell length as needed
                #lifted_topology = cell_cycle_lifting.lift_topology(data_geometric)

                #print("Lifted Topology: ", lifted_topology.keys())
                #print("Lifted Topology incidence: ", lifted_topology["incidence_1"])
                #print("Lifted Topology incidence: ", lifted_topology["incidence_2"])
                

                # Step 7: Cycle Sampling for Cell Construction
                


                # pooled_embeddings = embeddings[knn_indices.long()].mean(axis=1, keepdim=True).squeeze()

                # include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
                # inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
                # straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())
                


                # mask = torch.zeros((num_nodes, num_nodes),device=data.x.device)

                # node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,3))

                # incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix
                # incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')

                data_for_lifting={}

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