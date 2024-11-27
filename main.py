import argparse

import torch
from dataset.dataset_handler import choose_dataset
from model.GNN import GNN
from model.tnn_with_lifiting import TNN_KNN_MLP
from utils import parse_args, set_seed
import torch.nn as nn

train_losses = []
test_accuracies = []
train_accuracies = []
triangle_counts = []  # Add this list to store triangle counts


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    set_seed(args.seed)
    data, num_features, num_classes = choose_dataset(args)

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
        
        model.triangle_count = 0

        out = model(data.x, data.edge_index,data.edge_index_undirected ,data.batch)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item() , model.triangle_count
    

    def test():
        model.eval()
        out = model(data.x, data.edge_index, data.edge_index_undirected , data.batch)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum().item()
        test_accuracy = correct / data.test_mask.sum().item()
        train_correct = (pred[data.train_mask] == data.y[data.train_mask]).sum().item()
        train_accuracy = train_correct / data.train_mask.sum().item()
        return test_accuracy, train_accuracy

 
    for epoch in range(1, 100):
        train_loss,triangle_count = train()
        test_acc,train_acc = test()

        # Store the metrics
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        train_accuracies.append(train_acc)

        triangle_counts.append(triangle_count)  # Store triangle count
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name} gradient: {param.grad}")

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_acc:.4f}\n")
    
    import matplotlib.pyplot as plt

    # Save Train Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_loss.png", dpi=300)
    plt.close()

    # Save Train and Test Accuracies plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Train Accuracy", color="green")
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label="Test Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Train and Test Accuracies Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("train_test_accuracies.png", dpi=300)
    plt.close()
    # Save Triangle Counts plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(triangle_counts) + 1), triangle_counts, label="Triangles Considered", color="purple")
    plt.xlabel("Epochs")
    plt.ylabel("Number of Triangles")
    plt.title("Number of Triangles Considered Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("triangle_counts.png", dpi=300)
    plt.close()

