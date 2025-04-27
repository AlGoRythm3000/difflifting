"""HyperGat Layer."""

import torch

from layers.hypergnns.hypergat_layer import HyperGATLayer
from layers.unignns.unitgat_layer import UniGATLayer


class UniGAT(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads=8,
        n_layers=2,
        layer_drop=0.2,
        **kwargs,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            UniGATLayer(
                in_channels=in_channels if i == 0 else heads * hidden_channels,
                hidden_channels=hidden_channels,
                **kwargs,
            )
            for i in range(n_layers)
        )
        self.layer_drop = torch.nn.Dropout(layer_drop)
        self.lin0 = torch.nn.Linear(in_features=heads * hidden_channels, out_features=hidden_channels)
    def forward(self, x_0, incidence_1):
        for layer in self.layers:
            x_0, x_1 = layer.forward(x_0, incidence_1)
            x_0 = self.layer_drop(x_0)
        x_0 = self.lin0(x_0)
        return x_0, x_1
