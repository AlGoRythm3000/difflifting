import torch
from torch import nn
from torch_geometric.nn import global_mean_pool
from model.models.model_factory import ModelFactory
from tools.normalize import normalize_matrix


class TNN(nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, out_channels, normalize_laplacians=True,device="cpu", **kwargs):
        super().__init__()
        self.base_model = ModelFactory.create_model(model_type, in_channels, in_channels, in_channels, device, **kwargs)

        print("Type of base_model:", type(self.base_model))
        self.linear = nn.Linear(hidden_channels, out_channels)
        self.pooling_fun = global_mean_pool
        self.normalize_laplacians = normalize_laplacians
        self.model_type = model_type

    def forward(self, data):
        model_out = {}
        x=[]
        if self.model_type == "SCN2":
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            normalize_matrix(data.hodge_laplacian_0, 0),
                            normalize_matrix(data.hodge_laplacian_1, 1),
                            normalize_matrix(data.hodge_laplacian_2, 2))
        elif self.model_type == "CWN":
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            data.hodge_laplacian_1,
                            data.incidence_2,
                            data.incidence_1)
        elif self.model_type == "CXN":
            x = self.base_model(data.x_0, data.x_1,
                            data.laplacian_up_0,
                            data.incidence_2)

        model_out["x_0"] = x[0]
        model_out["x_1"] = x[1]
        model_out["x_2"] = x[2]
        return model_out
