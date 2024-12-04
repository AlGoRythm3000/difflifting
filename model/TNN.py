from topomodelx.nn.cell.cwn import CWN
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

        self.pooling_fun = global_mean_pool

    def forward(self, batch):
        model_out = {}
        x = self.base_model(batch.x_0, batch.x_1, batch.x_2,
                            batch.hodge_laplacian_0,
                            batch.hodge_laplacian_1,
                            batch.hodge_laplacian_2)
        model_out["x_0"] = x[0]
        model_out["x_1"] = x[1]
        model_out["x_2"] = x[2]
        return model_out