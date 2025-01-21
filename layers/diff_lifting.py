# import torch.nn
# import torch_geometric.utils
# from networkx.linalg.graphmatrix import incidence_matrix
# from toponetx import SimplicialComplex
# from torch_geometric.nn import global_mean_pool
#
# from dataset.dataset_handler import remove_duplicated_edges
# from model.GNN import GNN
# from preprocessing.preprocessing import remove_duplicate_edges
#
# from torch_geometric.data import Data, Batch
#
# from tools.feature_lifting.projection_sum import ProjectionSum
# from torch import nn
#
# class DiffLifting(torch.nn.Module):
#
#     def __init__(self, args,in_channels,num_classes,  mlp_hidden_dim, k):
#         super(DiffLifting, self).__init__()
#
#
#     def forward(self, data):
