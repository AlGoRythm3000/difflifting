import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

import torch_geometric

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

            mask_knn= torch.nn.functional.one_hot(data.batch_0,num_classes=vertex_slice.shape[0]-1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings) # Find if there is cdist without sqrt

            distances= mask_knn * distances + (1-mask_knn)*1e7
            #print(distances.shape)
            knn_indices = torch.topk(-distances, self.k, dim=-1)[1] 
            #print(knn_indices[:,0:3])
            # for i in range(vertex_slice[:-1].shape[0]):
            #     distances = torch.cdist(embeddings[vertex_slice[i]: vertex_slice[i+1], :], embeddings[vertex_slice[i]: vertex_slice[i+1], :])
            #     knn_indices[vertex_slice[i]: vertex_slice[i+1]] = vertex_slice[i] + torch.topk(-distances, self.k, dim=-1)[1]

            pooled_embeddings = embeddings[knn_indices.long()].mean(axis=1, keepdim=True).squeeze()

            include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
            inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
            straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())
            
            num_edges = edge_index_undirected.size(1)

            incidence_matrix_1 = torch.zeros((data.x.size(0), num_edges), device=data.x.device)

            for idx, edge in enumerate(edge_index_undirected.T):
                incidence_matrix_1[edge[0], idx] = 1
                incidence_matrix_1[edge[1], idx] = 1
            
            #print(straight_through_samples.repeat)

            #print(straight_through_samples.flatten())

            #print(knn_indices.flatten())

            # node_triangle_matrix = torch.zeros((data.x.size(0), data.x.size(0)), device=data.x.device)

            # triangle_mask = straight_through_samples.view(-1) == 1.0
            # selected_knn_indices = knn_indices[triangle_mask]  # Nós incluídos
            # straight_through_indices = list(torch.where(straight_through_samples==1)[0])

            # incidence_matrix_2 = torch.zeros((edge_index_undirected.shape[1], embeddings.shape[0]),
            #                                   device=data.x.device, requires_grad=True)
            num_nodes= data.x.size(0)

         

            mask = torch.zeros((num_nodes, num_nodes),device=data.x.device)
            # node_triangle_matrix= mask.scatter_(1, torch.cat((knn_indices.flatten().unsqueeze(1), torch.arange(0,num_nodes).repeat(3,1).T.flatten().unsqueeze(1)),axis=1), straight_through_samples)


            node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,3))

            #print("node_triangle: ", node_triangle_matrix)

            #print("node_triangle sum: ", node_triangle_matrix.sum(1))

            #print("grad: ", node_triangle_matrix.grad_fn)

            #mask.scatter_(0, knn_indices, straight_through_samples)
            
            #incidence_matrix_temp_2[knn_indices] = straight_through_samples

            #incidence_matrix_2 = incidence_matrix_temp_2.clone().requires_grad_()
            incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix

            incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')
            
            #print("incidence_matrix_2: ", incidence_matrix_2)

            #print("incidence_matrix_2 sum: ", incidence_matrix_2.sum(1))

            

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