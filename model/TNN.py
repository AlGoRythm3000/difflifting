from topomodelx.nn.hypergraph.unigcn import UniGCN
from topomodelx.nn.simplicial.sccnn import SCCNN
from topomodelx.nn.simplicial.scn2 import SCN2
from torch import nn
import torch
from topomodelx.nn.simplicial.san import SAN
from torch_geometric.nn import global_mean_pool

from tools.normalize import normalize_matrix


class TNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.base_model = SCN2(in_channels, in_channels, in_channels, n_layers=2)
        self.linear = nn.Linear(hidden_channels, out_channels)
        # IF GRAPH CLSASIFICATION
        self.pooling_fun = global_mean_pool

    def forward(self, data):
        x = self.base_model(data.x, data.x_1, data.x_2,
                            data.hodge_laplacian_0,
                            data.hodge_laplacian_1,
                            data.hodge_laplacian_2)
        return x