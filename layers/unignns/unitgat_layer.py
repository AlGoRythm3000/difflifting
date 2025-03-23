import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter
from torch_scatter import scatter
from torch_geometric.utils import softmax


class UniGATLayer(torch.nn.Module):
    """Layer of UniGAT.

    Attention-based implementation of a UniGNN-style layer.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features (per head).
    heads : int, default=8
        Number of attention heads.
    dropout : float, default=0.0
        Dropout applied to attention coefficients.
    negative_slope : float, default=0.2
        Negative slope for LeakyReLU in attention mechanism.
    use_bn : bool, default=False
        Whether to use batch normalization.
    use_norm : bool, default=False
        Whether to apply L2 normalization to output.
    skip_sum : bool, default=False
        Whether to apply skip connection (residual sum).
    **kwargs : optional
        Additional arguments.

    References
    ----------
    .. [1] Veličković et al., "Graph Attention Networks", ICLR 2018.
        https://arxiv.org/abs/1710.10903
    .. [2] Huang and Yang, "UniGNN: a unified framework for graph and hypergraph neural networks", IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 8,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        use_bn: bool = False,
        use_norm: bool = False,
        skip_sum: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = heads * hidden_channels
        self.heads = heads
        self.use_bn = use_bn
        self.use_norm = use_norm
        self.skip_sum = skip_sum

        self.W = nn.Linear(in_channels, self.out_channels, bias=False)
        self.att_v = Parameter(torch.Tensor(1, heads, hidden_channels))
        self.att_e = Parameter(torch.Tensor(1, heads, hidden_channels))

        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_drop = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(self.out_channels) if use_bn else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_v)
        nn.init.xavier_uniform_(self.att_e)
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""Forward pass of the UniGAT layer.

        The layer operates as follows:

        .. math::
            \begin{align*}
            &🟥 \quad x_0^T = W \cdot h_x^{(0)} \rightarrow \text{reshape into heads}\\
            &🟧 \quad m_{z}^{(0 \rightarrow 1)} = \text{scatter}_{e} \left( h_{x \in e} \right)\\
            &🟨 \quad \alpha_e = \text{softmax}_{v} \left( \text{LeakyReLU}(a_v^T \cdot h_v + a_e^T \cdot h_e) \right)\\
            &🟩 \quad m_x^{(1 \rightarrow 0)} = \text{scatter}_{v} \left( \alpha \cdot h_e \right)
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels)
            Input node features.
        incidence_1 : torch.sparse.Tensor, shape = (n_nodes, n_edges)
            Sparse incidence matrix B_1 mapping nodes to hyperedges.

        Returns
        -------
        x_0 : torch.Tensor
            Updated node features.
        x_1 : torch.Tensor
            Updated hyperedge features.
        """
        if x_0.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Mismatch in number of nodes in features and nodes: {x_0.shape[-2]} and {incidence_1.shape[-2]}."
            )

        vertex, edges = incidence_1.coalesce().indices()  # (2, nnz)

        N = x_0.size(0)
        H, C = self.heads, self.hidden_channels

        x_proj = self.W(x_0).view(N, H, C)  # (N, H, C)
        x_ve = x_proj[vertex]  # (nnz, H, C)

        # Aggregate from nodes to hyperedges
        x_1 = scatter(x_ve, edges, dim=0, reduce="mean")  # (n_edges, H, C)

        # Attention computation
        alpha_e = (x_1 * self.att_e).sum(-1)  # (n_edges, H)
        alpha_ev = alpha_e[edges]  # (nnz, H)
        alpha = self.leaky_relu(alpha_ev)
        alpha = softmax(alpha, vertex, num_nodes=N)  # (nnz, H)
        alpha = self.attn_drop(alpha).unsqueeze(-1)  # (nnz, H, 1)

        # Aggregate back to nodes with attention
        x_ev = x_1[edges] * alpha  # (nnz, H, C)
        x_0 = scatter(x_ev, vertex, dim=0, reduce='sum', dim_size=N)  # (N, H, C)
        x_0 = x_0.view(N, H * C)  # (N, out_channels)

        if self.use_bn:
            x_0 = self.bn(x_0)
        if self.use_norm:
            x_0 = F.normalize(x_0, p=2, dim=-1)

        return x_0, x_1
