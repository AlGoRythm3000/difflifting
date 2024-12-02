import torch.nn
import torch_geometric.utils

from dataset.dataset_handler import remove_duplicated_edges
from preprocessing.preprocessing import remove_duplicate_edges
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, Batch



class ProjectionSum(BaseTransform):
    r"""Lift r-cell features to r+1-cells by projection."""

    def __init__(self, **kwargs):
        super().__init__()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def lift_features(self, data):
        r"""Project r-cell features of a graph to r+1-cell structures."""
        keys = sorted(
            [key.split("_")[1] for key in data if ("incidence" in key and "-" not in key)]
        )
        for elem in keys:
            if f"x_{elem}" not in data:
                idx_to_project = 0 if elem == "hyperedges" else int(elem) - 1
                data["x_" + elem] = torch.matmul(
                    abs(data["incidence_" + elem].t()), data[f"x_{idx_to_project}"]
                )
        return data

    def forward(self, data):
        r"""Apply the lifting to the input data."""
        data = self.lift_features(data)
        return data
class DiffLifting(torch.nn.Module):

    def __init__(self, gnn, pool, mlp, k):
        super(DiffLifting, self).__init__()
        self.gnn = gnn
        self.pool = pool
        self.k = k
        self.mlp = mlp
        self.triangle_count = 0
        self.projection_sum = ProjectionSum()

    def forward(self, data):
        # edge_index_undirected, vertex_slices, edge_slices, batch = remove_duplicate_edges(data)
        # edge_index_undirected = remove_duplicate_edges(data.edge_index)
        edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)
        embeddings = self.gnn(data.x, data.edge_index)
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

            selected_embedding = straight_through_sample * pooled_embedding + (1 - straight_through_sample) * embeddings[i]

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

        node_edge_matrix = torch.zeros((data.x.size(0), num_edges), device=data.x.device)
        for idx, edge in enumerate(edge_index_undirected.T):
            node_edge_matrix[edge[0], idx] = 1
            node_edge_matrix[edge[1], idx] = 1

        # Apply ProjectionSum to lift node features to edge features
        data_for_lifting = {
            "x_0": embeddings,  # Node features
            "incidence_1": node_edge_matrix,  # Node-to-edge incidence matrix
            "incidence_2": incidence_matrix, #edge_to-triangle
        }
        lifted_data = self.projection_sum(data_for_lifting)

        # laplacian_up = incidence_matrix @ incidence_matrix.T
        # laplacian_down = node_edge_matrix.T @ node_edge_matrix

        data.x_1 = lifted_data["x_1"]
        data.x_2 = lifted_data["x_2"]

        new_edge_index, new_edge_attr = torch_geometric.utils.get_laplacian(data.edge_index)
        laplacian_0 = torch.sparse_coo_tensor(
                        indices=new_edge_index,
                        values=new_edge_attr,
                        size=(data.x.shape[0], data.x.shape[0])
                    )

        data.laplacian_up_0 = laplacian_0
        data.laplacian_down_0 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)

        data.laplacian_up_1 = torch.spmm(data_for_lifting["incidence_1"].T, data_for_lifting["incidence_1"])
        data.laplacian_down_1 = torch.spmm(data.laplacian_up_0, data.laplacian_up_0.T)

        data.laplacian_up_2 = torch.spmm(data_for_lifting["incidence_2"].T, data_for_lifting["incidence_2"])
        data.laplacian_down_2 = torch.spmm(data.laplacian_up_1, data.laplacian_up_1.T)
        data.node_edge_matrix = node_edge_matrix

        data.incidence_1 = data_for_lifting.get("incidence_1")
        data.incidence_2 = data_for_lifting.get("incidence_2")

        data.hodge_laplacian_0 = data.laplacian_up_0  + data.laplacian_down_0
        data.hodge_laplacian_1 = data.laplacian_up_1  + data.laplacian_down_1
        data.hodge_laplacian_2 = data.laplacian_up_2  + data.laplacian_down_2

        return data