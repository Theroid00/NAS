"""Proxy evaluation with explicit candidate failures and fatal infrastructure errors."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from training.config import LEARNING_RATE, WEIGHT_DECAY


def validate(model, loader, device, return_metrics=False):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    confusion = None
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite validation logits")
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if return_metrics:
                loss_sum += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                classes = logits.shape[1]
                if confusion is None:
                    confusion = torch.zeros((classes, classes), dtype=torch.long, device=device)
                confusion += torch.bincount(y * classes + logits.argmax(1), minlength=classes * classes).reshape(classes, classes)
    if total == 0:
        raise ValueError("Validation loader is empty")
    if return_metrics:
        from training.metrics import classification_metrics
        return {"accuracy": correct / total, "loss": loss_sum / total, "samples": total,
                **classification_metrics(confusion.cpu().numpy())}
    return correct / total


def evaluate_trial(chromosome, device="cpu", proxy_epochs=20, seed=42, split_seed=42,
                   proxy_size=None, max_params=None, dataset="breast_cancer_wisconsin", validation_size=None):
    from ga.chromosome import decode
    from models.mlp import build_model, count_parameters, estimate_parameters
    from data.tabular import get_proxy_loaders
    from utils.reproducibility import set_training_seed
    if proxy_epochs < 1:
        raise ValueError("Proxy epochs must be positive")
    set_training_seed(seed)
    started = time.perf_counter()
    model = None
    arch = decode(chromosome)
    from data.specs import dataset_spec
    spec = dataset_spec(dataset)
    params = estimate_parameters(arch, spec["input_features"], spec["num_classes"])
    try:
        if max_params is not None and params > max_params:
            return {"fitness": 0.0, "status": "parameter_limit", "num_params": params,
                    "elapsed_s": time.perf_counter() - started}
        model = build_model(arch, spec["input_features"], spec["num_classes"]).to(device)
        model.eval()
        with torch.no_grad():
            model(torch.zeros(2, spec["input_features"], device=device))
        params = count_parameters(model)
        train_loader, val_loader = get_proxy_loaders(proxy_size=proxy_size, seed=split_seed, training_seed=seed, dataset=dataset, validation_size=validation_size)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=proxy_epochs)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(proxy_epochs):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}")
                loss.backward()
                optimizer.step()
            scheduler.step()
        metrics = validate(model, val_loader, device, return_metrics=True)
        return {"fitness": metrics["accuracy"], "validation_loss": metrics["loss"],
                "validation_metrics": metrics,
                "training_samples": len(train_loader.dataset), "validation_samples": metrics["samples"],
                "status": "ok", "num_params": params,
                "elapsed_s": time.perf_counter() - started}
    except (torch.cuda.OutOfMemoryError, FloatingPointError) as error:
        return {"fitness": 0.0, "status": "out_of_memory" if isinstance(error, torch.cuda.OutOfMemoryError) else "diverged",
                "error": str(error), "num_params": params, "elapsed_s": time.perf_counter() - started}
    finally:
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def evaluate_architecture(chromosome, device="cpu", proxy_epochs=20, **kwargs):
    """Compatibility wrapper; infrastructure/programming errors propagate."""
    return evaluate_trial(chromosome, device, proxy_epochs, **kwargs)["fitness"]
