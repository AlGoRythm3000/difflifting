import torch
from tqdm import tqdm


def train(loader, model, loss_fn, optimizer, device):
    model.train()
    train_losses = []
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out.squeeze().float(), batch.y.squeeze())
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, evaluator=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss = loss_fn(out.squeeze().float(), batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs  # Para grafos
        total_samples += batch.num_graphs  # Total de amostras

        if not isinstance(loss_fn, torch.nn.L1Loss):
            total_correct += (out.argmax(dim=-1) == batch.y.squeeze()).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    if evaluator is not None:
        metrics = evaluator.eval(
            {"y_pred": out, "y_true": batch.y}
        )  # Ajuste se necessário
        return avg_loss, metrics[evaluator.eval_metric]
    return avg_loss, accuracy