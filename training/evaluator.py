"""
training/evaluator.py
=====================
Proxy-trains an architecture for a small number of epochs and returns
validation accuracy as the fitness signal for the GA.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def validate(model: nn.Module, loader, device: str) -> float:
    """Return top-1 accuracy on the given data loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def evaluate_architecture(
    chromosome: list,
    device: str = "cpu",
    proxy_epochs: int = 5,
) -> float:
    """
    Proxy-train an architecture encoded by `chromosome` and return
    validation accuracy.

    Uses the proxy subset (10k samples) for fast evaluation during the GA
    search. Returns 0.0 on any build/training error (acts as infinite-cost
    penalty so the GA avoids broken chromosomes).

    Args:
        chromosome:    Integer-index chromosome list
        device:        PyTorch device string
        proxy_epochs:  Number of training epochs on the proxy subset

    Returns:
        Validation accuracy ∈ [0, 1]
    """
    from ga.chromosome import decode
    from models.builder import build_model
    from data.cifar import get_proxy_loaders

    try:
        arch = decode(chromosome)
        model = build_model(arch).to(device)

        train_loader, val_loader = get_proxy_loaders()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=proxy_epochs)

        # Initialise LazyLinear by doing one forward pass
        model.train()
        sample_x, _ = next(iter(train_loader))
        _ = model(sample_x[:2].to(device))

        for epoch in range(proxy_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                
                # Divergence check: if loss explodes to NaN, architecture is dead
                if torch.isnan(loss):
                    print(f"  [!] Divergence detected (NaN loss) at epoch {epoch}. Aborting.")
                    return 0.0
                    
                loss.backward()
                optimizer.step()
            scheduler.step()

        return validate(model, val_loader, device)

    except Exception as e:
        print(f"[evaluator] Error evaluating chromosome {chromosome}: {e}")
        return 0.0
