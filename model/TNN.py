from topomodelx.nn.cell.ccxn import CCXN
from topomodelx.nn.cell.cwn import CWN
from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer
from topomodelx.nn.hypergraph.unisage import UniSAGE
from topomodelx.nn.hypergraph.unigin import UniGIN
from topomodelx.nn.hypergraph.unigcn import UniGCN

from layers.hypergnns.hypergat import HyperGAT
from topomodelx.nn.simplicial.scn2 import SCN2
from torch import nn
from torch_geometric.nn import global_mean_pool

from layers.unignns.unigat import UniGAT
from tools.normalize import normalize_matrix


class TNN(nn.Module):
    def __init__(self, model_type, in_channels, hidden_channels, normalize_laplacians=True,n_layers=4,device="cpu", **kwargs):
        super().__init__()
        if model_type == "CWN":
            self.base_model = CWN(in_channels, in_channels, in_channels, hidden_channels, n_layers=n_layers, **kwargs).to(device)
        elif model_type == "SCN2":
            self.base_model = SCN2(in_channels, in_channels, in_channels, n_layers=n_layers, **kwargs).to(device)
        elif model_type == "CXN":
            self.base_model =  CCXN(in_channels, in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "UniGCNII":
            self.base_model =  UniGCNII(in_channels, in_channels).to(device)
        elif model_type == "UniSAGE":
            self.base_model =  UniSAGE(in_channels, in_channels).to(device)
        elif model_type == "UniGAT":
            self.base_model =  UniGAT(in_channels, in_channels).to(device)
        elif model_type == "UniGCN":
            self.base_model = UniGCN(in_channels, in_channels).to(device)
        elif model_type == "HyperGAT":
            self.base_model = HyperGAT(in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "UniGIN":
            self.base_model = UniGIN(in_channels, in_channels, n_layers=n_layers).to(device)
        elif model_type == "AST":
            self.base_model =  AllSetTransformer(in_channels, in_channels,  n_layers=n_layers, n_heads=4).to(device)
        self.incidence_models = ["UniGCN", "HyperGAT", "UniGIN", "UniSAGE"]
        print("Type of base_model:", type(self.base_model))
        self.pooling_fun = global_mean_pool
        self.normalize_laplacians = normalize_laplacians
        self.model_type = model_type

    def forward(self, data):
        model_out = {}
        x=[]
        if self.model_type == "SCN2":
            print("SCN2")
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            normalize_matrix(data.hodge_laplacian_0, 0),
                            normalize_matrix(data.hodge_laplacian_1, 1),
                            normalize_matrix(data.hodge_laplacian_2, 2))
        elif self.model_type == "CWN":
            x = self.base_model(data.x_0, data.x_1, data.x_2,
                            data.adjacency_1,
                            data.incidence_2,
                            data.incidence_1.T)
        elif self.model_type == "CXN":
            x = self.base_model(data.x_0, data.x_1,
                            data.adjacency_0,
                            data.incidence_2.T)

        elif self.model_type == "AST":
            x = self.base_model(data.x_0, data.incidence_1)

            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out
        elif self.model_type in self.incidence_models:

            x = self.base_model(data.x_0, data.incidence_1)

            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out

        elif self.model_type == "UniGCNII":

            
            x = self.base_model(data.x_0, data.incidence_1)
            
            model_out["x_0"] = x[0]
            model_out["x_1"] = x[1]

            return model_out


        model_out["x_0"] = x[0]
        model_out["x_1"] = x[1]
        model_out["x_2"] = x[2]
        return model_out
    

"""UniGCNII class."""

import math

import torch

from topomodelx.nn.hypergraph.unigcnii_layer import UniGCNIILayer



class UniGCNII(torch.nn.Module):
    """Hypergraph neural network utilizing the UniGCNII layer [1]_ for node-level classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default=2
        Number of UniGCNII message passing layers.
    alpha : float, default=0.5
        Parameter of the UniGCNII layer.
    beta : float, default=0.5
        Parameter of the UniGCNII layer.
    input_drop : float, default=0.2
        Dropout rate for the input features.
    layer_drop : float, default=0.2
        Dropout rate for the hidden features.
    use_norm : bool, default=False
        Whether to apply row normalization after every layer.
    **kwargs : optional
        Additional arguments for the inner layers.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        alpha=0.5,
        beta=0.5,
        input_drop=0.2,
        layer_drop=0.2,
        use_norm=False,
        **kwargs,
    ):
        super().__init__()
        layers = []

        self.input_drop = torch.nn.Dropout(input_drop)
        self.layer_drop = torch.nn.Dropout(layer_drop)

        self.initial_linear_layer = torch.nn.Linear(in_channels, hidden_channels)

        for i in range(n_layers):
            beta = math.log(alpha / (i + 1) + 1)
            layers.append(
                UniGCNIILayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    alpha=alpha,
                    beta=beta,
                    use_norm=use_norm,
                    **kwargs,
                )
            )

        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x_0, incidence_1):
        """Forward pass through the model.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels)
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = (num_nodes, num_edges)
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        #print("FIRST X_0", x_0)
        x_0 = self.input_drop(x_0)
        x_0 = self.initial_linear_layer(x_0)
        x_0 = torch.nn.functional.relu(x_0)
        x_0_skip = x_0
        #print("FORWARD X_0", x_0)
        #assert(False)
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1, x_0_skip)
            #print("first",x_0, "\n\n ")
            x_0 = self.layer_drop(x_0)
            #print("second",x_0, "\n\n ")
            x_0 = torch.nn.functional.relu(x_0)
            #print("last",x_0, "\n\n ")

        return x_0, x_1



"""UniGCNII layer implementation."""
import torch

from topomodelx.base.conv import Conv



class UniGCNIILayer(torch.nn.Module):
    r"""
    Implementation of the UniGCNII layer [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    alpha : float
        The alpha parameter determining the importance of the self-loop (\theta_2).
    beta : float
        The beta parameter determining the importance of the learned matrix (\theta_1).
    use_norm : bool, default=False
        Whether to apply row normalization after the layer.
    **kwargs : optional
        Additional arguments for the layer modules.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        alpha: float,
        beta: float,
        use_norm=False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.linear = torch.nn.Linear(in_channels, hidden_channels, bias=False)
        self.conv = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            with_linear_transform=False,
        )
        self.use_norm = use_norm


    def reset_parameters(self) -> None:
        """Reset the parameters of the layer."""
        self.linear.reset_parameters()



    def forward(self, x_0, incidence_1, x_skip=None):
        r"""Forward pass of the UniGCNII layer.

        The forward pass consists of:
        - two messages, and
        - a skip connection with a learned update function.

        First every hyper-edge sums up the features of its constituent edges:

        .. math::
            \begin{align*}
            & 🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)} = (B^T_1)\_{zy} \cdot h^{t,(0)}_y \\
            & 🟧 \quad m_z^{(0\rightarrow1)} = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}
            \end{align*}

        Second, the second message is normalized with the node and edge degrees:

        .. math::
            \begin{align*}
            & 🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)} \\
            & 🟧 \quad m_{x}^{(1\rightarrow0)}  = \frac{1}{\sqrt{d_x}}\sum_{z \in \mathcal{C}(x)} \frac{1}{\sqrt{d_z}}m_{z \rightarrow x}^{(1\rightarrow0)} \\
            \end{align*}

        Third, the computed message is combined with skip connections and a linear transformation using hyperparameters alpha and beta:

        .. math::
            \begin{align*}
            & 🟩 \quad m_x^{(0)}  = m_x^{(1 \rightarrow 0)} \\
            & 🟦 \quad m_x^{(0)}  = ((1-\beta)I + \beta W)((1-\alpha)m_x^{(0)} + \alpha \cdot h_x^{t,(0)}) \\
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels)
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = (num_nodes, num_edges)
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.
        x_skip : torch.Tensor, shape = (num_nodes, in_channels)
            Original node features of the hypergraph used for the skip connections.
            If not provided, the input to the layer is used as a skip connection.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        x_skip = x_0 if x_skip is None else x_skip
        incidence_1_transpose = incidence_1.transpose(0, 1)

        # First message without any learning or parameters
        x_1 = self.conv(x_0, incidence_1_transpose)

        # Compute node and edge degrees for normalization.
        node_degree = torch.sum(incidence_1.to_dense(), dim=1)

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8  # Small constant to prevent division by zero
        node_degree = node_degree + epsilon  # Add epsilon to node degrees

        # Average node degree for each edge.
        edge_degree = torch.sum(torch.diag(node_degree) @ incidence_1, dim=0)

        # Add epsilon to edge degrees as well to prevent division by zero
        edge_degree = edge_degree + epsilon

        # Second message normalized with node and edge degrees (using broadcasting)
        x_0 = (1 / torch.sqrt(node_degree).unsqueeze(-1)) * self.conv(
            x_1, incidence_1 @ torch.diag(1 / torch.sqrt(edge_degree))
        )

        # Introduce skip connections with hyperparameter alpha and beta
        x_combined = ((1 - self.alpha) * x_0) + (self.alpha * x_skip)
        x_0 = ((1 - self.beta) * x_combined) + self.beta * self.linear(x_combined)

        if self.use_norm:
            rownorm = x_0.detach().norm(dim=1, keepdim=True)
            scale = rownorm.pow(-1)
            scale[torch.isinf(scale)] = 0.0
            x_0 = x_0 * scale

        return x_0, x_1


