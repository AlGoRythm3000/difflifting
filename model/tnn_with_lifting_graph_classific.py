import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, DeepSetsAggregation
import torch.nn.functional as F
from torch_geometric.utils import degree


import torch_geometric

from layers.deepset import DeepSetLayer
from layers.diff_lifting import DiffLifting
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


        self.triangle_count = 0  # Add this to track triangles
        self.diff_lifting = diff_lifting
        self.tnn_type = tnn_type
        self.num_classes = num_classes
        #self.lin= nn.Linear(tnn_out_feat, num_classes)
        self.feature_encoder = AllCellFeatureEncoder(in_channels=[in_channels, in_channels, in_channels], out_channels=hidden_dim, proj_dropout=0.5)
        if diff_lifting:
            self.deep_set_layers = []
            for i in range(k):
                self.deep_set_layers.append(DeepSetLayer(in_channels, hidden_dim))
            if args.gnn == "GIN":
                self.gnn = GIN(in_channels, hidden_dim, embedding_dim, num_layers_gnn).to(device)
            elif args.gnn == "GPS":
                self.gnn = GPS(in_channels, hidden_dim, args.positional_walking_len , num_layers_gnn).to(device)
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

            mask_knn= torch.nn.functional.one_hot(data.batch_0,num_classes=vertex_slice.shape[0]-1)

            mask_knn = mask_knn.float() @ mask_knn.T.float()

            distances = torch.cdist(embeddings, embeddings) # Find if there is cdist without sqrt

            distances= mask_knn * distances + (1-mask_knn)*1e7
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



            mask = torch.zeros((num_nodes, num_nodes),device=data.x.device)

            node_triangle_matrix= mask.scatter_(1, knn_indices, straight_through_samples.repeat(1,3))

            incidence_matrix_2= incidence_matrix_1.T @ node_triangle_matrix
            incidence_matrix_2= torch.div(incidence_matrix_2,2,rounding_mode='trunc')

            data_for_lifting={}

            if self.tnn_type == "UniGCNII":
                incidence_matrix_1 = torch.cat((incidence_matrix_1, node_triangle_matrix), dim=1)
                data_for_lifting = {
                    "x_0": x.float(),  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                }

            else:
                data_for_lifting = {
                    "x_0": x.float(),  # Node features
                    "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
                    "incidence_2": incidence_matrix_2,  # edge_to-triangle
                }


            lifted_data = self.projection_sum(data_for_lifting)

            data.x_0 = x.float()

            if self.tnn_type != "UniGCNII":
                data = self.__create_laplacians(data, incidence_matrix_1, lifted_data, data_for_lifting)


            data.incidence_1 = data_for_lifting.get("incidence_1")
            data.incidence_1= torch.Tensor(data.incidence_1).to_sparse_coo()


        data = self.feature_encoder(data)
        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)


        return out["logits"]