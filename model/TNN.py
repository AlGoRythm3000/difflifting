from torch import nn 
import torch
from topomodelx.nn.simplicial.san import SAN
from torch_geometric.nn import global_mean_pool


class TNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.base_model = SAN(in_channels, hidden_channels, n_layers=2)
        self.linear = nn.Linear(hidden_channels, out_channels)
        # IF GRAPH CLSASIFICATION
        self.pooling_fun = global_mean_pool

    def forward(self, x, laplacian_up, laplacian_down, node_edge_matrix,batch):
        x = self.base_model(x, laplacian_up, laplacian_down)
        tnn_output = self.pooling_fun(torch.sparse.mm(node_edge_matrix, x), batch)
        return torch.sigmoid(self.linear(tnn_output))