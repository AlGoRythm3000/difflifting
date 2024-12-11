import torch.nn
import torch_geometric.utils
from networkx.linalg.graphmatrix import incidence_matrix
from toponetx import SimplicialComplex

from dataset.dataset_handler import remove_duplicated_edges
from preprocessing.preprocessing import remove_duplicate_edges

from torch_geometric.data import Data, Batch

from tools.feature_lifting.projection_sum import ProjectionSum


class DiffLifting(torch.nn.Module):

    def __init__(self, gnn, pool, mlp, k):
        super(DiffLifting, self).__init__()
        self.gnn = gnn
        self.pool = pool
        self.k = k
        self.mlp = mlp
        self.triangle_count = 0
        self.projection_sum = ProjectionSum()


    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        edge_index_undirected, vertex_slice, new_slices, data.batch = remove_duplicate_edges(data)
        embeddings = self.gnn(x, edge_index)
        distances = torch.cdist(embeddings, embeddings)
        knn_indices = torch.topk(-distances, self.k, dim=-1)[1]
        num_edges = edge_index_undirected.size(1)
        # max_triangles = embeddings.size(0)


        edge_map = {tuple(sorted(edge)): idx for idx, edge in enumerate(edge_index_undirected.T.tolist())}

        num_nodes = embeddings.size(0)
        knn_set = torch.cat((
            embeddings[knn_indices],  # Shape: [num_nodes, k, emb_dim]
            embeddings.unsqueeze(1)  # Shape: [num_nodes, 1, emb_dim]
        ), dim=1)  # Shape: [num_nodes, k+1, emb_dim]

        # 2. Realize o pooling em batch
        pooled_embeddings = self.pool(knn_set, batch=None)
        pooled_embeddings = pooled_embeddings.view(num_nodes, -1)  # Shape: [num_nodes, emb_dim]

        # 3. Calcule as probabilidades de inclusão e as amostras
        include_probs = torch.sigmoid(self.mlp(pooled_embeddings))  # Shape: [num_nodes, 1]
        inclusion_samples = (torch.rand_like(include_probs) < include_probs).float()
        straight_through_samples = inclusion_samples + (include_probs - include_probs.detach())

        # 4. Atualize os embeddings selecionados vetorizadamente
        # selected_embeddings = (
        #         straight_through_samples * pooled_embeddings
        #         + (1 - straight_through_samples) * embeddings
        # )

        # 5. Atualize a matriz de triângulos vetorizada
        triangle_mask = straight_through_samples.view(-1) == 1.0
        selected_knn_indices = knn_indices[triangle_mask]  # Nós incluídos

        incidence_matrix_2 = torch.zeros((edge_index_undirected.shape[1], selected_knn_indices.shape[0]), device=data.x.device)

        incidence_matrix_1 = torch.zeros((data.x.shape[0], edge_index_undirected.shape[1]), device=data.x.device)

        # for i in range(edge_index_undirected.shape[1]):
        #     for j in range(selected_knn_indices.shape[0]):
        #         if set(edge_index_undirected[:, i].cpu().numpy()).issubset(set(selected_knn_indices[j].cpu().numpy())):
        #             incidence_matrix_2[i, j] = 1.0

        edges_sorted = torch.sort(edge_index_undirected.T, dim=1)[0]  # Shape: [num_edges, 2]
        triangles_sorted = torch.sort(selected_knn_indices, dim=1)[0]  # Shape: [num_triangles, 3]

        # Expandir as dimensões para comparação
        edges_expanded = edges_sorted.unsqueeze(1)  # Shape: [num_edges, 1, 2]
        triangles_expanded = triangles_sorted.unsqueeze(0)  # Shape: [1, num_triangles, 3]

        # Verificar se cada nó da aresta está no triângulo
        matches = (edges_expanded.unsqueeze(-1) == triangles_expanded.unsqueeze(
            -2))  # Shape: [num_edges, num_triangles, 2, 3]

        # Verificar se ambos os nós da aresta estão presentes no triângulo
        edge_in_triangle = matches.any(dim=-1).all(dim=-1)  # Shape: [num_edges, num_triangles]

        # Converter para float para formar a matriz de incidência
        incidence_matrix_2 = edge_in_triangle.float()  # Shape: [num_edges, num_triangles]

        for idx, edge in enumerate(edge_index_undirected.T):
            incidence_matrix_1[edge[0], idx] = 1
            incidence_matrix_1[edge[1], idx] = 1

        # Apply ProjectionSum to lift node features to edge features
        data_for_lifting = {
            "x_0": x.float(),  # Node features
            "incidence_1": incidence_matrix_1,  # Node-to-edge incidence matrix
            "incidence_2": incidence_matrix_2, #edge_to-triangle
        }
        lifted_data = self.projection_sum(data_for_lifting)

        data.x_0 = x.float()
        data.x_1 = lifted_data["x_1"]
        data.x_2 = lifted_data["x_2"]

        new_edge_index, new_edge_attr = torch_geometric.utils.get_laplacian(data.edge_index)
        laplacian_0 = torch.sparse_coo_tensor(
                        indices=new_edge_index,
                        values=new_edge_attr,
                        size=(data.x.shape[0], data.x.shape[0])
                    )

        data.laplacian_up_0 = laplacian_0
        # data.laplacian_down_0 = torch.zeros((data.x.size(0), num_edges), device=data.x.device).to_sparse_coo()

        data.laplacian_up_1 = torch.spmm(data_for_lifting["incidence_2"], data_for_lifting["incidence_2"].T).to_sparse_coo()
        data.laplacian_down_1 = torch.spmm(data_for_lifting["incidence_1"].T, data_for_lifting["incidence_1"]).to_sparse_coo()

        data.laplacian_down_2 = torch.spmm(data_for_lifting["incidence_2"].T, data_for_lifting["incidence_2"]).to_sparse_coo()
        data.node_edge_matrix = incidence_matrix_1

        data.incidence_1 = data_for_lifting.get("incidence_1")
        data.incidence_2 = data_for_lifting.get("incidence_2")

        data.hodge_laplacian_0 = data.laplacian_up_0  #+ data.laplacian_down_0
        data.hodge_laplacian_1 = data.laplacian_up_1  + data.laplacian_down_1
        data.hodge_laplacian_2 =  data.laplacian_down_2

        return data