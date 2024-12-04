import os.path as osp

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader, Batch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.utils import degree
from torch_geometric.datasets import ZINC, TUDataset
import torch_geometric.transforms as T
from torch_geometric.datasets import KarateClub
from torch_geometric.datasets import Planetoid

from tools.collate import collate_fn
from tools.lifting.clique_lifting import SimplicialCliqueLifting
from tools.lifting.khop import SimplicialKHopLifting
from tools.lifting.neighboorhood_complex import NeighborhoodComplexLifting
from tools.normalize import normalize_matrix

NODES_PREDICTION_DATASET = ["CORA", "CITESEER", "PUBMED", "KARATECLUB"]
LIFTINGS = {
    "SimplicialCliqueLifting":SimplicialCliqueLifting,
    "NeighborhoodComplexLifting": NeighborhoodComplexLifting,
    "SimplicialKHopLifting":SimplicialKHopLifting,
}

class FilterConstant(object):
  def __init__(self, dim):
    """Initializes the FilterConstant class.

    Parameters
    ----------
    dim : int
        The number of features to output for each node.
    """
    
    self.dim = dim

  def __call__(self, data):
    """Replace node features with a constant vector of ones.

    Parameters
    ----------
    data : torch_geometric.data.Data
        The input data object.

    Returns
    -------
    torch_geometric.data.Data
        The modified data object with node features replaced by a constant vector of ones.
    """
    data.x = torch.ones(data.num_nodes, self.dim)
    return data


def get_ogb_data(name: str) -> PygGraphPropPredDataset:
    """Loads the OGB dataset specified by name and ensures features are float.

    Args:
        name (str): The name of the OGB dataset to load.

    Returns:
        PygGraphPropPredDataset: The loaded dataset object.
    """
    path = osp.dirname(osp.realpath(__file__))
    dataset = PygGraphPropPredDataset(name=name, root=path)



    return dataset
def get_data_loaders(train_set, val_set, test_set, batch_size):
    """Returns three DataLoaders from the given datasets.

    Args:
        train_set: The dataset to use for the training DataLoader.
        val_set: The dataset to use for the validation DataLoader.
        test_set: The dataset to use for the testing DataLoader.
        batch_size: The batch size to use for the training DataLoader.

    Returns:
        A tuple containing the DataLoaders for training, validation, and testing.
    """
    from torch.utils.data import DataLoader

    num_workers: int = 0,
    pin_memory: bool = False,
    train_loader = DataloadDataset(
        train_set
    )
    train_loader = DataLoader(
        train_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = DataloadDataset(
        val_set
    )
    valid_loader = DataLoader(
        valid_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataloadDataset(
        test_set
    )
    test_loader = DataLoader(
        test_loader,
        batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    return train_loader, valid_loader, test_loader


def divide_train_val_test_split(dataset: PygGraphPropPredDataset, batch_size):
    """Returns three DataLoaders for training, validation, and testing from the given dataset.

    Args:
        dataset: The PygGraphPropPredDataset to use.
        batch_size: The batch size to use for the training DataLoader.

    Returns:
        A tuple containing the DataLoaders for training, validation, and testing.
    """
    if dataset.name.startswith("ogbg"):
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(
            dataset[split_idx["train"]], batch_size=batch_size, shuffle=True
        )

        valid_loader = DataLoader(
            dataset[split_idx["valid"]],
            batch_size=split_idx["valid"].shape[0],
            shuffle=False,
        )
        test_loader = DataLoader(
            dataset[split_idx["test"]],
            batch_size=split_idx["test"].shape[0],
            shuffle=False,
        )
        return train_loader, valid_loader, test_loader


def get_graph_classification_dataset(dataset: str, batch_size, args, seed=42):
    """Returns DataLoaders for the given dataset.

    Args:
        dataset: The name of the dataset to use.
        batch_size: The batch size for the DataLoader.
        dim: The dimension of the node features. If None, the default is used.
        seed: The random seed for splitting the dataset.

    Returns:
        A tuple containing the DataLoaders for the training, validation, and testing sets.
    """
    if dataset.startswith("ogbg"):
        dataset = get_ogb_data(dataset)
        train_loader, val_loader, test_loader = divide_train_val_test_split(dataset, batch_size)
        dataloaders = (train_loader, val_loader, test_loader)

    elif dataset == "ZINC":
        train_set, val_set, test_set = get_zinc()
        dataloaders = get_data_loaders(train_set, test_set, val_set, batch_size)
        return  dataloaders, train_set.num_node_features, 1
    else:
        dataset = tu_datasets(dataset, args)
        train_set, val_set, test_set = data_split(dataset, seed)
        dataloaders = get_data_loaders(train_set, test_set, val_set, batch_size)

    return dataloaders, dataset.num_features, dataset.num_classes


def ensure_float_features(dataset):
    """
    Converts all features in the dataset to float tensors.

    Args:
        dataset (torch_geometric.data.Dataset): The dataset to process.

    Returns:
        torch_geometric.data.Dataset: The dataset with float features.
    """
    for data in dataset:
        if hasattr(data, 'x') and data.x is not None:
            data.x = data.x.long()
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr.long()
    return dataset

def get_zinc(args):
    """Loads the ZINC dataset and returns the training, validation, and test sets.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), "..", "ZINC")
    train_data = ZINC(path, subset=True, split="train")
    data_val = ZINC(path, subset=True, split="val")
    data_test = ZINC(path, subset=True, split="test")

    if args.lifting != "diffLifting":
        train_data =  lift_topology(train_data, args)
        data_val = lift_topology(data_val, args)
        data_test = lift_topology(data_test, args)
    return train_data, data_val, data_test


def tu_datasets(name,args, no_feat_replacement='constant'):
    """Loads a TUDataset and applies feature replacement if necessary.

    Args:
        name (str): The name of the TU dataset to load.
        no_feat_replacement (str, optional): How to replace missing node features.
            Defaults to 'constant'. Options:
            - 'constant': Replace with a constant vector.
            - 'degree': Replace with one-hot encoding of node degrees.

    Returns:
        TUDataset: The loaded dataset, potentially with transformed features.
    """
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', name)
    dataset = TUDataset(name=name, root=path, pre_transform=None,)
    if not hasattr(dataset, 'x'):
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
        max_degree = max(max_degree, degs[-1].max().item())
        if no_feat_replacement == 'constant':
            dataset.transform = FilterConstant(10)
        elif no_feat_replacement == 'degree':
            T.OneHotDegree(max_degree)
    if args.lifting != "diffLifting":
        return lift_topology(dataset, args)
    return dataset


def lift_topology(dataset, args):
        data_list = []
        max_dim = 0
        for i, d in enumerate(dataset):
            lift_fn = LIFTINGS[args.lifting]()
            new_data = lift_fn(d)
            for key, value in new_data.items():
                if key.startswith("hodge_laplacian_"):
                    setattr(d, key,normalize_matrix(value, int(key[-1])))
                setattr(d, key,value)
            data_list.append(d)
        dataset.data, dataset.slices = dataset.collate(data_list)
        return dataset

def data_split(dataset, seed):
    """Splits a dataset into training, validation, and test sets using stratified shuffle split.

    Args:
        dataset: The dataset to be split.
        seed: The random seed to ensure reproducibility.

    Returns:
        A tuple containing the training, validation, and test datasets.
    """
    skf_train = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_test_idx = list(skf_train.split(torch.zeros(len(dataset)), dataset.y))[0]
    skf_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx, test_idx = list(skf_val.split(torch.zeros(val_test_idx.size), dataset.y[val_test_idx]))[0]
    train_data = dataset[train_idx]
    val_data = dataset[val_test_idx[val_idx]]
    test_data = dataset[val_test_idx[test_idx]]
    return train_data, val_data, test_data

def remove_duplicated_edges(edge_index):
    """Removes duplicated edges from an edge_index tensor.

    Args:
        edge_index: A tensor of shape (2, num_edges) containing the edges of a graph.

    Returns:
        A tensor of shape (2, num_edges) containing the edges of the graph without duplicates.
    """
    arestas = set()
    for i in range(edge_index.size(1)):
        u = edge_index[0, i].item()
        v = edge_index[1, i].item()
        arestas.add((min(u, v), max(u, v)))  # Armazenar como um par ordenado
    return torch.tensor(list(arestas), dtype=torch.long).T


def get_node_prediction_dataset(dataset, dim=None, seed=42):
    """Loads a dataset for node-level prediction tasks.

    Args:
        dataset (str): The name of the dataset to load. Options: KARATECLUB, Cora, CiteSeer, PubMed.
        dim (int, optional): The dimension of the node features. Defaults to None.
        seed (int, optional): The random seed for splitting the dataset. Defaults to 42.

    Returns:
        A tuple containing the DataLoaders for the training, validation, and testing sets.
    """
    if dataset == "KARATECLUB":
        dataset = KarateClub()
        data = dataset[0]
        num_train_nodes = int(0.8 * data.num_nodes)
        data.train_mask = torch.zeros(data.num_nodes, dtype=bool)
        data.train_mask[:num_train_nodes] = True
        data.test_mask = ~data.train_mask
        data.edge_index_undirected= remove_duplicated_edges(data.edge_index)
        
    elif dataset=="CORA":
        dataset = Planetoid(root='data', name='cora')
        data = dataset[0]
        data.edge_index_undirected= remove_duplicated_edges(data.edge_index)
        
    elif dataset=="CITESEER":
        dataset = Planetoid(root='data', name='CiteSeer')
        data = dataset[0]
        data.edge_index_undirected= remove_duplicated_edges(data.edge_index)

    elif dataset=="PUBMED":
        dataset = Planetoid(root='data', name='pubmed')
        data = dataset[0]        
        data.edge_index_undirected= remove_duplicated_edges(data.edge_index)


    return data, dataset.num_features, dataset.num_classes

def choose_dataset(args):
    """Chooses the appropriate dataset function based on the input data.

    Args:
        data (str): The name of the dataset to load.

    Returns:
        A function that loads the dataset. The function is either
        `get_node_prediction_dataset` or `get_graph_classification_dataset`.
    """
    if args.dataset in NODES_PREDICTION_DATASET:
        return get_node_prediction_dataset(args.dataset)
    else:
        return get_graph_classification_dataset(args.dataset, args.batch_size, args)


import torch_geometric


class DataloadDataset(torch_geometric.data.Dataset):
    """Custom dataset to return all the values added to the dataset object.

    Parameters
    ----------
    data_lst : list[torch_geometric.data.Data]
        List of torch_geometric.data.Data objects.
    """

    def __init__(self, data_lst):
        super().__init__()
        self.data_lst = data_lst

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.data_lst)})"

    def get(self, idx):
        """Get data object from data list.

        Parameters
        ----------
        idx : int
            Index of the data object to get.

        Returns
        -------
        tuple
            Tuple containing a list of all the values for the data and the corresponding keys.
        """
        data = self.data_lst[idx]
        keys = list(data.keys())
        return ([data[key] for key in keys], keys)

    def len(self):
        """Return the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data_lst)
