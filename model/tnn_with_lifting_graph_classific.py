import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import torch_sparse
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F

from layers.diff_lifting import DiffLifting
from model.TNN import TNN
from torch_geometric.transforms import BaseTransform

from preprocessing.preprocessing import remove_duplicate_edges



class TNN_KNN_MLP_G(nn.Module):
    def __init__(self,in_channels, gnn, mlp_hidden_dim, tnn_hidden_dim, num_classes, k=2, rank=2):
        super(TNN_KNN_MLP_G, self).__init__()
        self.gnn = gnn
        self.k = k
        self.pool = global_mean_pool
        self.triangle_count = 0  # Add this to track triangles

        self.mlp = nn.Sequential(
            nn.Linear(gnn.out_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_hidden_dim, 1),
        )
        self.classifier = nn.Linear(gnn.out_channels, num_classes)
        self.diff_lifting = DiffLifting(self.gnn, self.pool, self.mlp, self.k)
        self.tnn = TNN(
            in_channels=in_channels,
            hidden_channels=tnn_hidden_dim,
            out_channels=num_classes,
        )
        self.classifier = nn.Sequential(
            nn.Linear(3*in_channels, num_classes),
        )
        # self.projection_sum = ProjectionSum()

    def forward(self, batch):
        data = batch
        graphs_lifted = []
        data = self.diff_lifting(data)
        # tnn_output = self.tnn(data.x_1, laplacian_up=data.laplacian_up.to_sparse(),
        #                       laplacian_down=data.laplacian_down.to_sparse(),node_edge_matrix=data.node_edge_matrix, batch=batch)
        tnn_output = self.tnn(data)
        #Redout Model
        pool_0 = self.pool(tnn_output[0], data.batch)
        pool_1 = self.pool(torch.spmm(data.node_edge_matrix,tnn_output[1]), data.batch)
        pool_2 = self.pool(torch.spmm(data.node_edge_matrix,torch.mm(data.incidence_2, tnn_output[2])), data.batch)
        out = torch.cat([pool_0, pool_1, pool_2], dim=1)
        out = self.classifier(out)

        return F.log_softmax(out, dim=-1)
