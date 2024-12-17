import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from layers.diff_lifting import DiffLifting
from model.GNN import GNN
from model.TNN import TNN
from torch_geometric.transforms import BaseTransform

from preprocessing.preprocessing import remove_duplicate_edges
from tools.redout import PropagateSignalDown


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
            embeddings = self.gnn(x, edge_index)
            distances = torch.cdist(embeddings, embeddings)
            knn_indices = torch.topk(-distances, self.k, dim=-1)[1]


            num_nodes = embeddings.size(0)
            knn_set = torch.cat((
                embeddings[knn_indices],  # Shape: [num_nodes, k, emb_dim]
                embeddings.unsqueeze(1)  # Shape: [num_nodes, 1, emb_dim]
            ), dim=1)  # Shape: [num_nodes, k+1, emb_dim]

            # 2. Realize pooling in batch
            pooled_embeddings = self.pool(knn_set, batch=None)
            pooled_embeddings = pooled_embeddings.view(num_nodes, -1)  # Shape: [num_nodes, emb_dim]

            # 3. Calculate the probability of inclusion samples
            include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
            inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
            straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())



            # 5. Update matrix of triangles (or cells)
            triangle_mask = straight_through_samples.view(-1) == 1.0
            selected_knn_indices = knn_indices[triangle_mask]  # Included Nodes

            incidence_matrix_2 = torch.zeros((edge_index_undirected.shape[1], selected_knn_indices.shape[0]))

            incidence_matrix_1 = torch.zeros((data.x.shape[0], edge_index_undirected.shape[1]), device=data.x.device)


            edges_sorted = torch.sort(edge_index_undirected.T, dim=1)[0]  # Shape: [num_edges, 2]
            triangles_sorted = torch.sort(selected_knn_indices, dim=1)[0]  # Shape: [num_triangles, 3]
            for i in range(triangles_sorted.shape[0]):
                for j in range(edges_sorted.shape[0]):
                    if set(edges_sorted[j]).issubset(set(triangles_sorted)):
                        incidence_matrix_2[j,i] = 1.0
            # # Expand edges for comparison
            # edges_expanded = edges_sorted.unsqueeze(1)  # Shape: [num_edges, 1, 2]
            # triangles_expanded = triangles_sorted.unsqueeze(0)  # Shape: [1, num_triangles, 3]
            #
            # # Verify if edge in triangle
            # matches = (edges_expanded.unsqueeze(-1) == triangles_expanded.unsqueeze(
            #     -2))  # Shape: [num_edges, num_triangles, 2, 3]
            #
            # # Verificar se ambos os nós da aresta estão presentes no triângulo
            # edge_in_triangle = matches.any(dim=-1).all(dim=-1)  # Shape: [num_edges, num_triangles]

            # Converter para float para formar a matriz de incidência
            # incidence_matrix_2 = edge_in_triangle.float()  # Shape: [num_edges, num_triangles]

            for idx, edge in enumerate(edge_index_undirected.T):
                incidence_matrix_1[edge[0], idx] = 1
                incidence_matrix_1[edge[1], idx] = 1

            # Apply ProjectionSum to lift node features to edge features
            data_for_lifting = {
                "x_0": x.float(),  # Node features
                "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                "incidence_2": incidence_matrix_2,  # edge_to-triangle
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
            data.node_edge_matrix = incidence_matrix_1

            data.incidence_1 = data_for_lifting.get("incidence_1")
            data.incidence_2 = data_for_lifting.get("incidence_2")

            data.hodge_laplacian_0 = data.laplacian_up_0  # + data.laplacian_down_0
            data.hodge_laplacian_1 = data.laplacian_up_1 + data.laplacian_down_1
            data.hodge_laplacian_2 = data.laplacian_down_2

        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)


        return out["logits"]