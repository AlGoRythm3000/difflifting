import torch
import torch.nn as nn
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

# from layers.diff_lifting import DiffLifting
from model.GNN import GNN
from model.TNN import TNN
from torch_geometric.transforms import BaseTransform

from model.tnn_with_lifiting import ProjectionSum
from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import PropagateSignalDown


class TNN_KNN_MLP_G(nn.Module):
    def __init__(self,in_channels, args, mlp_hidden_dim, tnn_hidden_dim, num_classes, k=2, diff_lifting=False,global_pool="sum",device="cpu", tnn_type= "SCN2"):
        super(TNN_KNN_MLP_G, self).__init__()
        self.k = k
        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        if diff_lifting:
            self.gnn = GNN(in_channels, args.hidden_dim, mlp_hidden_dim)
            self.pool = global_mean_pool
            self.k = k
            self.mlp = nn.Sequential(
                nn.Linear(mlp_hidden_dim, 2 * mlp_hidden_dim),
                nn.ReLU(),
                nn.Linear(2 * mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(mlp_hidden_dim, 1),
            )
            self.triangle_count = 0
            self.projection_sum = ProjectionSum()

        self.tnn = TNN(
            model_type=tnn_type,  # choose TNN model
            in_channels=in_channels,
            hidden_channels=tnn_hidden_dim,
            out_channels=num_classes,
            device=device
        )

        self.readout = PropagateSignalDown(**{
            "readout_name": "PropagateSignalDownLinear",
            "num_cell_dimensions": 3,
            "hidden_dim": in_channels,
            "out_channels": num_classes,
            "task_level": "graph",
            "pooling_type": global_pool,
        })

    def forward(self, batch):
        data = batch
        if self.diff_lifting:
            x, edge_index = data.x.float(), data.edge_index
            edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)
            embeddings = self.gnn(x, edge_index, data.batch)
            distances = torch.cdist(embeddings, embeddings)
            knn_indices = torch.topk(-distances, self.k, dim=-1)[1]
            num_edges = edge_index_undirected.size(1)
            max_triangles = embeddings.size(0)
            incidence_matrix_temp = torch.zeros(
                (num_edges, max_triangles), device=data.x.device
            )

            edge_map = {tuple(sorted(edge)): idx for idx, edge in enumerate(edge_index_undirected.T.tolist())}

            for i in range(embeddings.size(0)):
                knn_set = torch.cat((embeddings[knn_indices[i]], embeddings[i].unsqueeze(0)), dim=0)
                pooled_embedding = self.pool(knn_set, batch=None)

                include_prob = torch.sigmoid(self.mlp(pooled_embedding)).view(-1)
                inclusion_sample = (torch.rand_like(include_prob) < include_prob).float()
                straight_through_sample = inclusion_sample + (include_prob - include_prob.detach())

                selected_embedding = straight_through_sample * pooled_embedding + (1 - straight_through_sample) * \
                                     embeddings[i]

                if inclusion_sample.item() == 1.0:
                    self.triangle_count += 1  # Increment triangle count
                    complex_set = tuple(knn_indices[i].tolist())
                    node_indices = sorted(complex_set)
                    edges_in_triangle = [
                        (node_indices[0], node_indices[1]),
                        (node_indices[0], node_indices[2]),
                        (node_indices[1], node_indices[2]),
                    ]

                    # Populate incidence matrix with differentiable `straight_through_sample`
                    for edge in edges_in_triangle:
                        edge_idx = edge_map.get(tuple(sorted(edge)))
                        if edge_idx is not None:
                            incidence_matrix_temp[edge_idx, i] = straight_through_sample

            incidence_matrix = incidence_matrix_temp.clone().requires_grad_()
            incidence_matrix = incidence_matrix
            node_edge_matrix = torch.zeros((data.x.size(0), num_edges), device=data.x.device)
            for idx, edge in enumerate(edge_index_undirected.T):
                node_edge_matrix[edge[0], idx] = 1
                node_edge_matrix[edge[1], idx] = 1

            # Apply ProjectionSum to lift node features to edge features
            data_for_lifting = {
                "x_0": x.float(),  # Node features
                "incidence_1": node_edge_matrix,  # Node-to-edge incidence matrix
                "incidence_2": incidence_matrix,  # edge_to-triangle
            }
            lifted_data = self.projection_sum(data_for_lifting)

            data.x_0 = x.float()
            data.x_1 = lifted_data["x_1"]
            data.x_2 = lifted_data["x_2"]

            new_edge_index, new_edge_attr = torch_geometric.utils.get_laplacian(data.edge_index)
            laplacian_0 = torch.sparse_coo_tensor(
                indices=new_edge_index,
                values=new_edge_attr,
                size=(data.x.shape[0], data.x.shape[0])
            )

            data.laplacian_up_0 = laplacian_0
            # data.laplacian_down_0 = torch.zeros((data.x.size(0), num_edges), device=data.x.device).to_sparse_coo()

            data.laplacian_up_1 = torch.spmm(data_for_lifting["incidence_2"],
                                             data_for_lifting["incidence_2"].T).to_sparse_coo()
            data.laplacian_down_1 = torch.spmm(data_for_lifting["incidence_1"].T,
                                               data_for_lifting["incidence_1"]).to_sparse_coo()

            data.laplacian_down_2 = torch.spmm(data_for_lifting["incidence_2"].T,
                                               data_for_lifting["incidence_2"]).to_sparse_coo()
            data.node_edge_matrix = node_edge_matrix

            data.incidence_1 = data_for_lifting.get("incidence_1")
            data.incidence_2 = data_for_lifting.get("incidence_2")

            data.hodge_laplacian_0 = data.laplacian_up_0  # + data.laplacian_down_0
            data.hodge_laplacian_1 = data.laplacian_up_1 + data.laplacian_down_1
            data.hodge_laplacian_2 = data.laplacian_down_2

        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)


        return out["logits"]