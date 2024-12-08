import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, global_add_pool, global_mean_pool, GCNConv

from layers.gnns.gcn_layer import GcnCreator
from layers.gnns.gin_layer import GinCreator


class GNN(nn.Module):
    def __init__(
        self,
        gnn,
        hidden_dim,
        depth,
        num_node_features,
        out_channels,
        global_pooling,
        batch_norm=True,
    ):
        super().__init__()
        if gnn == "gin":
            gnn_instance = GinCreator(hidden_dim, batch_norm)
        elif gnn == "gcn":
            gnn_instance = GcnCreator(hidden_dim, batch_norm)

        build_gnn_layer = gnn_instance.return_gnn_instance
        if global_pooling == "mean":
            graph_pooling_operation = global_mean_pool
        elif global_pooling == "sum":
            graph_pooling_operation = global_add_pool

        self.pooling_fun = graph_pooling_operation
        self.embedding = torch.nn.Linear(num_node_features, hidden_dim)
        self.out_channels = out_channels
        layers = [build_gnn_layer(is_last=i == (depth - 1)) for i in range(depth)]

        self.layers = nn.ModuleList(layers)

        dim_before_class = hidden_dim
        self.classif = torch.nn.Sequential(
            nn.Linear(dim_before_class, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_channels),
        )

    def forward(self, x, edge_index):
        x = self.embedding(x.float())

        for layer in self.layers:
            x = layer(x, edge_index=edge_index)

        # x = self.pooling_fun(x, data.batch)
        x = self.classif(x)
        return x

