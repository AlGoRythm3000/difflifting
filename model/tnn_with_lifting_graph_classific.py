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
from tools.redout import PropagateSignalDown


class TNN_KNN_MLP_G(nn.Module):
    def __init__(self,in_channels, gnn, mlp_hidden_dim, tnn_hidden_dim, num_classes, k=2, diff_lifting=False,rank=2):
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
        self.diff_lifting = diff_lifting
        if diff_lifting:
            self.diff_lifting = DiffLifting(self.gnn, self.pool, self.mlp, self.k)
        self.tnn = TNN(
            in_channels=in_channels,
            hidden_channels=tnn_hidden_dim,
            out_channels=num_classes,
        )
        self.classifier = nn.Sequential(
            nn.Linear(3*in_channels, num_classes),
        )
        self.readout = PropagateSignalDown(**{
            "readout_name": "PropagateSignalDownLinear",
            "num_cell_dimensions": 3,
            "hidden_dim": in_channels,
            "out_channels": num_classes,
            "task_level": "graph",
            "pooling_type": "sum",
        })
        # self.projection_sum = ProjectionSum()

    def forward(self, batch):
        data = batch
        if self.diff_lifting:
            data = self.diff_lifting(data)
        # tnn_output = self.tnn(data.x_1, laplacian_up=data.laplacian_up.to_sparse(),
        #                       laplacian_down=data.laplacian_dn.to_sparse(),node_edge_matrix=data.node_edge_matrix, batch=batch)
        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)
        # out = self.classifier(pool_2)


        return F.log_softmax(out["logits"], dim=-1)
