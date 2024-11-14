from torch import nn 
import torch
from topomodelx.nn.simplicial.san import SAN


class TNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.base_model = SAN(in_channels, hidden_channels, n_layers=2)
        self.linear = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, laplacian_up, laplacian_down):
        x = self.base_model(x, laplacian_up, laplacian_down)
        return torch.sigmoid(self.linear(x))