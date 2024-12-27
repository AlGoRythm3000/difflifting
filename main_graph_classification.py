import argparse

import torch
from torch import tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ogb.graphproppred import Evaluator
from torchinfo import summary

from model.tnn_with_lifting_graph_classific import TNN_KNN_MLP_G
from train import train, evaluate
from dataset.dataset_handler import choose_dataset
from model.GNN import GNN

from utils import parse_args, set_seed
import torch.nn as nn
import os
train_losses = []
test_accuracies = []
train_accuracies = []
triangle_counts = []  # Add this list to store triangle counts

torch.autograd.set_detect_anomaly(True)
import tempfile
# import mlflow.pytorch
if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()
    # mlflow.set_experiment(f"/{args.tnn}_{args.lifting}_{args.seed}_{args.dataset}")
    set_seed(args.seed)
    data, num_features, num_classes = choose_dataset(args, device)
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]


    diff_lifting = True if args.lifting == "diffLifting" else False
    model = TNN_KNN_MLP_G(num_features, args, mlp_hidden_dim=args.hidden_dim, tnn_hidden_dim=args.hidden_dim, num_classes=num_classes,
                          k=3, diff_lifting=diff_lifting, global_pool=args.global_pooling, device=device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()



    def train_eval(model, train_loader, val_loader, test_loader, loss_fn, optimizer, evaluator, device):
        train_loss = train(train_loader, model, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, evaluator)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device, evaluator)
        return train_loss, val_loss, val_acc, test_loss, test_acc


    train_losses = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []

    print(
        "Number of parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        min_lr=1e-6,
        patience=args.lr_decay_patience,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    if args.dataset == "ZINC":
        loss_fn = torch.nn.L1Loss(reduction='mean')
    evaluator = None
    if args.dataset == "ogbg-molhiv":
        evaluator = Evaluator(args.dataset)

    # with mlflow.start_run() as run:
    #
    #     mlflow.log_params(args.__dict__,)
    #
    #
    #     mlflow.pytorch.log_model(model, 'models')
    #     with open("model_summary.txt", "w", encoding="utf-8") as f:
    #         f.write(str(summary(model, depth=5, verbose=2)))
    #     mlflow.log_artifact("model_summary.txt")
    for epoch in range(1, args.max_epochs):
        train_loss, val_loss, val_acc, test_loss, test_acc = train_eval(
            model,
            train_loader,
            val_loader,
            test_loader,
            loss_fn,
            optimizer,
            evaluator,
            device
        )
        # mlflow.log_metric('train loss',torch.tensor(train_loss).mean().item(), step=epoch)
        # mlflow.log_metric('val loss', val_loss.item(), step=epoch)
        # mlflow.log_metric('test loss', test_loss.item(), step=epoch)
        # mlflow.log_metric('test acc', test_acc.item(), step=epoch)
        # mlflow.pytorch.autolog()
        # for name, param in model.named_parameters():
        #     print(f"{name} gradient: {param.grad}")
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)  # test losses

        val_accuracies.append(val_acc)
        val_losses.append(val_loss)  # test losses

        train_losses.append(torch.tensor(train_loss).mean())  # train losses

        # if (epoch - 1) % args.interval == 0:
        print(
            f"{epoch:3d}: Train Loss: {torch.tensor(train_loss).mean():.3f},"
            f" Val Loss: {val_loss:.3f}, Val Acc: {val_accuracies[-1]:.3f}, "
            f"Test Loss: {test_loss:.3f}, Test Acc: {test_accuracies[-1]:.3f}"
        )

        scheduler.step(val_acc)

        if epoch > 2 and val_accuracies[-1] <= val_accuracies[-2 - epochs_no_improve]:
            epochs_no_improve = epochs_no_improve + 1

        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.early_stop_patience:
            print("Early stopping!")
            break

    results = {
        "train_losses": tensor(train_losses),
        "test_accuracies": tensor(test_accuracies),
        "test_losses": tensor(test_losses),
        "val_accuracies": tensor(val_accuracies),
        "val_losses": tensor(val_losses),
    }
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    torch.save(
        results, f"{args.logdir}/{args.dataset}_{args.lifting}_{args.gnn}_{args.tnn}_{args.seed}.results"
    )