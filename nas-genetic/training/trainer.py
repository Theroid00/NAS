"""
training/trainer.py
===================
Full training pipeline for the best architecture discovered by the GA.

Uses the complete 50k CIFAR-10 training set with cosine annealing LR decay,
early stopping on validation accuracy, and checkpoint saving.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from training.config import FULL_EPOCHS, FULL_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from training.evaluator import validate


def full_train(
    chromosome: list,
    device: str = "cpu",
    epochs: int = FULL_EPOCHS,
    save_dir: str = "experiments/best_architectures",
    run_id: str = "best",
) -> dict:
    """
    Train the best chromosome on the full CIFAR-10 dataset.

    Args:
        chromosome: Integer-index chromosome to train
        device:     PyTorch device string
        epochs:     Total training epochs
        save_dir:   Directory to save checkpoints and results
        run_id:     Identifier used in saved file names

    Returns:
        dict with keys: test_accuracy, val_accuracy, num_params, save_path
    """
    from ga.chromosome import decode
    from models.builder import build_model, count_parameters
    from data.cifar import get_full_loaders

    os.makedirs(save_dir, exist_ok=True)

    arch = decode(chromosome)
    model = build_model(arch).to(device)

    train_loader, val_loader, test_loader = get_full_loaders(batch_size=FULL_BATCH_SIZE)
    
    # 1. Switch to Adam to match GA search space optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Initialise LazyLinear
    model.train()
    sample_x, _ = next(iter(train_loader))
    _ = model(sample_x[:2].to(device))
    num_params = count_parameters(model)
    
    # 2. Add DataParallel for multi-GPU scaling
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    print(f"\nFull training | params={num_params:,} | device={device} | epochs={epochs}")

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(save_dir, f"ckpt_{run_id}_best.pt")
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()

        train_acc = correct / total
        val_acc = validate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_ckpt_path)

        elapsed = time.time() - t0
        history.append({"epoch": epoch + 1, "train_acc": train_acc, "val_acc": val_acc})
        print(
            f"  Epoch {epoch+1:3d}: train={train_acc:.4f}  val={val_acc:.4f}  "
            f"best_val={best_val_acc:.4f}  ({elapsed:.1f}s)"
        )

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_acc = validate(model, test_loader, device)

    print(f"\nFinal test accuracy : {test_acc:.4f}")
    print(f"Best val accuracy   : {best_val_acc:.4f}")

    results = {
        "chromosome": chromosome,
        "arch": arch,
        "num_params": num_params,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "history": history,
    }
    results_path = os.path.join(save_dir, f"full_train_{run_id}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    return results
