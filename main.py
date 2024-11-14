import argparse

import torch
from dataset.dataset_handler import choose_dataset
from model.GNN import GNN
from model.tnn_with_lifiting import TNN_KNN_MLP
from utils import parse_args, set_seed
import torch.nn as nn



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    set_seed(args.seed)
    data, num_features, num_classes = choose_dataset(args.dataset)
    data.to(device)
    print(data.edge_index_undirected)

    # raise Exception
    num_classes = data.y.max().item() + 1
    gnn = GNN(in_channels=num_features, hidden_channels=16, out_channels=8)
    model = TNN_KNN_MLP(gnn, mlp_hidden_dim=16, tnn_hidden_dim=16, num_classes=num_classes, k=3)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index,data.edge_index_undirected ,data.batch)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
    

    def test():
        model.eval()
        out = model(data.x, data.edge_index, data.edge_index_undirected , data.batch)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        test_accuracy = correct / data.test_mask.sum().item()
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
        train_accuracy = train_correct / data.train_mask.sum().item()
        return test_accuracy, train_accuracy

    train_losses = []
    test_accuracies = []
    train_accuracies = []

    for epoch in range(1, 500):
        train_loss = train()
        test_acc,train_acc = test()

        # Store the metrics
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_acc:.4f}\n")
