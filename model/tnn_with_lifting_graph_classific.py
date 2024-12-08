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
    def __init__(self,in_channels, gnn, mlp_hidden_dim, tnn_hidden_dim, num_classes, k=2, diff_lifting=False,global_pool="sum",device="cpu", tnn_type= "SCN2"):
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
            data = self.diff_lifting(data)

        tnn_output = self.tnn(data)
        out = self.readout(tnn_output, batch)


        return out["logits"]