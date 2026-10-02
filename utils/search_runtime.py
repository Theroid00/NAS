"""Shared budgets, isolated evaluation workers, and durable trial provenance."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import platform
import random
import time
from uuid import uuid4

from ga.chromosome import decode, SEARCH_SPACE, SCHEMA_VERSION


def validate_budget(budget, proxy_epochs, population_size=1, tournament_k=1):
    if any(type(v) is not int or v < 1 for v in (budget, proxy_epochs, population_size, tournament_k)):
        raise ValueError("Evaluation budget, epochs, population, and tournament size must be positive integers")
    if budget < population_size:
        raise ValueError("Evaluation budget must cover the initial population")


def resolve_devices(device, parallel=False):
    import torch
    parsed = torch.device(device)
    if parsed.type == "cpu":
        if parallel:
            raise ValueError("Parallel evaluation requires explicit available CUDA devices")
        return ["cpu"]
    if parsed.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"Requested device is unavailable: {device}")
    index = parsed.index if parsed.index is not None else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"CUDA index {index} exceeds available device count {count}")
    if parallel:
        if parsed.index is not None:
            raise ValueError("Use --device cuda --parallel to select all GPUs, or an indexed device without --parallel")
        if count < 2:
            raise ValueError("Parallel search requires at least two GPUs")
        return [f"cuda:{i}" for i in range(count)]
    return [f"cuda:{index}"]


def evaluate_task(chromosome, device, proxy_epochs, seed, split_seed, proxy_size, max_params):
    from training.evaluator import evaluate_trial
    return evaluate_trial(chromosome, device, proxy_epochs, seed=seed,
                          split_seed=split_seed, proxy_size=proxy_size, max_params=max_params)


class SearchSession:
    def __init__(self, method, config, log_dir, save_dir, smoke=False, evaluator=None):
        self.method, self.config, self.smoke = method, config, smoke
        self.evaluator = evaluator or evaluate_task
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]
        self.log_dir, self.save_dir = Path(log_dir), Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trial_path = self.log_dir / f"trials_{self.run_id}.jsonl"
        self.metadata_path = self.log_dir / f"metadata_{self.run_id}.json"
        self.count, self.best, self.executors = 0, None, []
        self.started = time.perf_counter()
        versions = {}
        for name in ("torch", "torchvision", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.metadata = {"run_id": self.run_id, "method": method, "schema_version": SCHEMA_VERSION,
                         "search_space": SEARCH_SPACE, "smoke": smoke, "config": config,
                         "python": platform.python_version(), "platform": platform.platform(),
                         "versions": versions, "status": "running"}

    def _write_metadata(self):
        self.metadata_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def __enter__(self):
        self.devices = [self.config["device"]] if self.smoke or self.config.get("injected_evaluator") else resolve_devices(
            self.config["device"], self.config.get("parallel", False))
        self.metadata["devices"] = self.devices
        self._write_metadata()
        self.file = self.trial_path.open("x", encoding="utf-8")
        try:
            if not self.smoke and not self.config.get("injected_evaluator"):
                # Fail on dataset/infrastructure errors before spending the search budget.
                from data.cifar import get_proxy_loaders
                get_proxy_loaders(proxy_size=self.config["proxy_size"], seed=self.config["split_seed"])
            if len(self.devices) > 1:
                self.executors = [ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
                                  for _ in self.devices]
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def _seed(self, trial_id):
        raw = f"{self.config['seed']}:{trial_id}".encode()
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")

    def evaluate(self, population, generation=None):
        if self.count + len(population) > self.config["evaluation_budget"]:
            raise ValueError("Evaluation would exceed the declared budget")
        tasks = []
        for offset, chromosome in enumerate(population):
            decode(chromosome)
            trial_id = self.count + offset + 1
            device = self.devices[(trial_id - 1) % len(self.devices)]
            args = (list(chromosome), device, self.config["proxy_epochs"], self._seed(trial_id),
                    self.config["split_seed"], self.config["proxy_size"], self.config.get("max_params"))
            if self.executors:
                pending = self.executors[(trial_id - 1) % len(self.executors)].submit(self.evaluator, *args)
            else:
                pending = None
            tasks.append((trial_id, args, pending))
        fitnesses = []
        for trial_id, args, pending in tasks:
            chromosome, device, epochs, seed, split_seed, proxy_size, max_params = args
            t0 = time.perf_counter()
            try:
                if self.smoke:
                    outcome = {"fitness": random.Random(seed).uniform(0.1, 0.9), "status": "smoke", "elapsed_s": 0.0}
                else:
                    outcome = pending.result() if pending else self.evaluator(*args)
                if isinstance(outcome, (int, float)):
                    outcome = {"fitness": outcome, "status": "ok"}
                fit = outcome["fitness"]
                if not isinstance(fit, (int, float)) or not math.isfinite(fit) or not 0 <= fit <= 1:
                    raise ValueError(f"Invalid trial fitness: {fit!r}")
            except BaseException as error:
                self._record({"trial_id": trial_id, "generation": generation, "chromosome": chromosome,
                              "arch": decode(chromosome), "seed": seed, "device": device,
                              "status": "error", "fitness": None, "error": str(error),
                              "elapsed_s": time.perf_counter() - t0})
                raise
            record = {**outcome, "trial_id": trial_id, "generation": generation,
                      "chromosome": chromosome, "arch": decode(chromosome), "seed": seed,
                      "device": device, "split_seed": split_seed, "proxy_size": proxy_size,
                      "proxy_epochs": epochs, "elapsed_s": outcome.get("elapsed_s", time.perf_counter() - t0)}
            self._record(record)
            if record["status"] in ("ok", "smoke") and (self.best is None or fit > self.best["fitness"]):
                self.best = record
            fitnesses.append(fit)
        return fitnesses

    def _record(self, record):
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()
        self.count += 1

    def finish(self, log_path=None):
        if self.best is None:
            raise RuntimeError("No valid candidate was evaluated; inspect trial failures")
        result = {"run_id": self.run_id, "method": self.method, "schema_version": SCHEMA_VERSION,
                  "search_space": SEARCH_SPACE, "smoke": self.smoke,
                  "winner_policy": "best_observed_validation_accuracy",
                  "best_chromosome": self.best["chromosome"], "best_arch": self.best["arch"],
                  "best_fitness": self.best["fitness"], "best_trial_id": self.best["trial_id"],
                  "hyperparams": self.config, "evaluation_count": self.count,
                  "elapsed_s": time.perf_counter() - self.started,
                  "trial_path": str(self.trial_path), "metadata_path": str(self.metadata_path),
                  "log_path": log_path}
        path = self.save_dir / f"best_{self.run_id}.json"
        result["save_path"] = str(path)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def __exit__(self, kind, error, traceback):
        for executor in self.executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self.file.close()
        self.metadata.update(status="failed" if error else "completed", evaluation_count=self.count,
                             elapsed_s=time.perf_counter() - self.started)
        if error:
            self.metadata["error"] = str(error)
        self._write_metadata()
