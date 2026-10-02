"""Shared Adam training and validation-based checkpoint selection for every method."""
import json
from pathlib import Path
import time
import torch
from torch import nn, optim
from tqdm import tqdm
from training.config import FULL_EPOCHS, FULL_BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY
from training.evaluator import validate
from utils.reproducibility import set_training_seed
from utils.search_runtime import resolve_devices


def full_train(chromosome=None, device="cpu", epochs=FULL_EPOCHS,
               save_dir="experiments/best_architectures", run_id="best", seed=42,
               split_seed=42, evaluate_test=True, baseline=False, parallel=False):
    from ga.chromosome import decode, SCHEMA_VERSION
    from models.builder import build_model
    from models.baselines.resnet import build_baseline_resnet
    if epochs < 1:
        raise ValueError("Full-training epochs must be positive")
    if any(type(value) is not int or not 0 <= value < 2 ** 32 for value in (seed, split_seed)):
        raise ValueError("Training and split seeds must be integers within 0..2^32-1")
    arch = None if baseline else decode(chromosome)
    set_training_seed(seed)
    model = build_baseline_resnet() if baseline else build_model(arch)
    return train_model(model, device, epochs, save_dir, run_id, seed, split_seed,
                       evaluate_test, parallel, {"chromosome": chromosome, "arch": arch,
                                                 "baseline": baseline, "schema_version": SCHEMA_VERSION})


def train_model(model, device, epochs, save_dir, run_id, seed, split_seed,
                evaluate_test=True, parallel=False, metadata=None):
    from models.builder import count_parameters
    from data.cifar import get_full_loaders
    if epochs < 1:
        raise ValueError("Full-training epochs must be positive")
    devices = resolve_devices(device, parallel)
    primary = devices[0]
    model = model.to(primary)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, 3, 32, 32, device=primary))
    params = count_parameters(model)
    if parallel:
        model = nn.DataParallel(model, device_ids=[int(d.split(":")[1]) for d in devices])
    train_loader, val_loader, test_loader = get_full_loaders(batch_size=FULL_BATCH_SIZE,
                                                          seed=split_seed, training_seed=seed)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    root = Path(save_dir)
    root.mkdir(parents=True, exist_ok=True)
    # A retrain must never silently overwrite a previous result/checkpoint.
    if (root / f"full_train_{run_id}.json").exists() or (root / f"ckpt_{run_id}_best.pt").exists():
        run_id += "_" + str(time.time_ns())
    checkpoint = root / f"ckpt_{run_id}_best.pt"
    best_val = -1.0
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        running_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x, y = x.to(primary), y.to(primary)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite full-training loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            running_loss += loss.item() * y.size(0)
        if total == 0:
            raise ValueError("Training loader is empty")
        scheduler.step()
        val = validate(model, val_loader, primary)
        if val > best_val:
            best_val = val
            underlying = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(underlying.state_dict(), checkpoint)
        history.append({"epoch": epoch + 1, "train_acc": correct / total,
                        "train_loss": running_loss / total, "val_acc": val})
        print(f"Epoch {epoch + 1}: train={correct / total:.4f}, val={val:.4f}, best={best_val:.4f}")
    underlying = model.module if isinstance(model, nn.DataParallel) else model
    underlying.load_state_dict(torch.load(checkpoint, map_location=primary, weights_only=True))
    test_acc = validate(model, test_loader, primary) if evaluate_test else None
    results_path = root / f"full_train_{run_id}.json"
    result = {**(metadata or {}), "run_id": run_id, "seed": seed, "split_seed": split_seed,
              "num_params": params, "best_val_accuracy": best_val, "test_accuracy": test_acc,
              "history": history, "elapsed_s": time.perf_counter() - started,
              "checkpoint_path": str(checkpoint.resolve()), "results_path": str(results_path.resolve()),
              "protocol": {"optimizer": "Adam", "learning_rate": LEARNING_RATE,
                           "weight_decay": WEIGHT_DECAY, "epochs": epochs, "batch_size": FULL_BATCH_SIZE,
                           "train_samples": len(train_loader.dataset), "validation_samples": len(val_loader.dataset),
                           "checkpoint_policy": "best_validation", "devices": devices}}
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_checkpoint(results_path, device="cpu"):
    """Explicit final test evaluation after protocol/candidate choices are frozen."""
    from models.builder import build_model
    from models.baselines.resnet import build_baseline_resnet
    from data.cifar import get_full_loaders
    path = Path(results_path)
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("test_accuracy") is not None:
        raise ValueError("This checkpoint already has a recorded test evaluation")
    primary = resolve_devices(device)[0]
    set_training_seed(result["seed"])
    model = build_baseline_resnet() if result.get("baseline") else build_model(result["arch"])
    model = model.to(primary)
    model.load_state_dict(torch.load(result["checkpoint_path"], map_location=primary, weights_only=True))
    _, _, test = get_full_loaders(seed=result["split_seed"], training_seed=result["seed"])
    result["test_accuracy"] = validate(model, test, primary)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
