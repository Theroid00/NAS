"""Shared budgets, single-device evaluation, and durable trial provenance."""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import time
import subprocess
from uuid import uuid4

from ga.chromosome import decode, SEARCH_SPACE, SCHEMA_VERSION
from utils.persistence import atomic_json, RunLock

RESUME_VERSION = 1


def validate_budget(budget, proxy_epochs, population_size=1, tournament_k=1):
    if any(type(v) is not int or v < 1 for v in (budget, proxy_epochs, population_size, tournament_k)):
        raise ValueError("Evaluation budget, epochs, population, and tournament size must be positive integers")
    if budget < population_size:
        raise ValueError("Evaluation budget must cover the initial population")


def resolve_device(device):
    import torch
    parsed = torch.device(device)
    if parsed.type == "cpu":
        return "cpu"
    if parsed.type != "cuda" or not torch.cuda.is_available():
        raise ValueError(f"Requested device is unavailable: {device}")
    index = parsed.index if parsed.index is not None else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise ValueError(f"CUDA index {index} exceeds available device count {count}")
    return f"cuda:{index}"


def evaluate_task(chromosome, device, proxy_epochs, seed, split_seed, proxy_size, max_params, dataset, validation_size):
    from training.evaluator import evaluate_trial
    return evaluate_trial(chromosome, device, proxy_epochs, seed=seed,
                          split_seed=split_seed, proxy_size=proxy_size, max_params=max_params, dataset=dataset, validation_size=validation_size)


class SearchSession:
    def __init__(self, method, config, log_dir, save_dir, smoke=False, evaluator=None, resume=None):
        for name in ("seed", "split_seed"):
            if type(config[name]) is not int or not 0 <= config[name] < 2 ** 32:
                raise ValueError(f"{name} must be an integer within 0..2^32-1")
        self.method, self.config, self.smoke = method, config, smoke
        self.evaluator = evaluator or evaluate_task
        self.resume = Path(resume).resolve() if resume else None
        self.replay = []
        self.previous_elapsed = 0.0
        self.run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:10]
        self.log_dir, self.save_dir = Path(log_dir), Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.trial_path = self.log_dir / f"trials_{self.run_id}.jsonl"
        self.metadata_path = self.log_dir / f"metadata_{self.run_id}.json"
        if self.resume:
            saved = json.loads(self.resume.read_text(encoding="utf-8"))
            self._check_resume(saved)
            self.run_id = saved["run_id"]
            self.metadata_path = self.resume
            self.trial_path = Path(saved["trial_path"])
            self.log_dir = self.trial_path.parent
            self.save_dir = Path(saved["save_dir"])
        self.count, self.best = 0, None
        self.evaluation_seconds = 0.0
        self.started = time.perf_counter()
        versions = {}
        for name in ("torch", "scikit-learn", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        self.metadata = {"run_id": self.run_id, "method": method, "schema_version": SCHEMA_VERSION,
                         "dataset_name": self.config["dataset"], "search_space": SEARCH_SPACE, "smoke": smoke, "config": config,
                         "python": platform.python_version(), "platform": platform.platform(),
                         "versions": versions, "status": "running"}
        self.metadata.update(resume_version=RESUME_VERSION, trial_path=str(self.trial_path.resolve()),
                             save_dir=str(self.save_dir.resolve()))
        try:
            root = Path(__file__).resolve().parents[1]
            revision = subprocess.run(["git", "-c", f"safe.directory={root.as_posix()}", "rev-parse", "HEAD"],
                                      cwd=root, capture_output=True, text=True, check=True)
            self.metadata["git_commit"] = revision.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            self.metadata["git_commit"] = None

    def _check_resume(self, saved):
        if (saved.get("resume_version") != RESUME_VERSION or saved.get("schema_version") != SCHEMA_VERSION
                or saved.get("search_space") != SEARCH_SPACE):
            raise ValueError("Run is incompatible with the current resumable search format")
        if saved["method"] != self.method or saved["config"] != self.config or saved["smoke"] != self.smoke:
            raise ValueError("Resume requires the original method and search configuration")

    def _load_trials(self):
        raw = self.trial_path.read_bytes()
        offset = 0
        for line in raw.splitlines(keepends=True):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if offset + len(line) != len(raw) or line.endswith(b"\n"):
                    raise ValueError("Trial journal is corrupt; refusing to resume")
                # Preserve an interrupted write before recovering the complete prefix.
                backup = self.trial_path.with_name(self.trial_path.name + ".partial." + uuid4().hex)
                backup.write_bytes(line)
                with self.trial_path.open("r+b") as stream:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                break
            offset += len(line)
            if record["status"] == "error":
                if record["trial_id"] != len(self.replay) + 1:
                    raise ValueError("Invalid failed trial sequence")
                self.evaluation_seconds += record.get("elapsed_s", 0.0)
                continue
            if record["trial_id"] != len(self.replay) + 1 or len(self.replay) >= self.config["evaluation_budget"]:
                raise ValueError("Invalid completed trial sequence")
            fit = record["fitness"]
            if not isinstance(fit, (int, float)) or not math.isfinite(fit) or not 0 <= fit <= 1:
                raise ValueError("Invalid archived trial fitness")
            decode(record["chromosome"])
            if record.get("arch") != decode(record["chromosome"]):
                raise ValueError("Archived architecture does not match its chromosome")
            if record["status"] not in ("ok", "smoke", "parameter_limit", "out_of_memory", "diverged"):
                raise ValueError("Unknown archived trial status")
            self.replay.append(record)
        # A complete final JSON record without its newline must not join the next record.
        if raw and not raw.endswith(b"\n") and offset == len(raw):
            with self.trial_path.open("ab") as stream:
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _write_metadata(self):
        atomic_json(self.metadata_path, self.metadata)

    def __enter__(self):
        self.lock = RunLock(self.metadata_path.with_suffix(".lock"))
        self.lock.acquire()
        self.file = None
        try:
            return self._enter_locked()
        except BaseException:
            if self.file is not None:
                self.file.close()
            self.lock.release()
            raise

    def _enter_locked(self):
        self.device = self.config["device"] if self.smoke or self.config.get("injected_evaluator") else resolve_device(self.config["device"])
        if self.resume:
            saved = json.loads(self.resume.read_text(encoding="utf-8"))
            self._check_resume(saved)
            current_revision = self.metadata["git_commit"]
            current_versions = self.metadata["versions"]
            if current_versions != saved["versions"]:
                raise ValueError("Dependency versions changed; use the original environment to resume")
            self.metadata = saved
            self.previous_elapsed = saved.get("elapsed_s", 0.0)
            self.metadata.setdefault("resume_history", []).append({"git_commit": current_revision,
                                                                   "versions": current_versions,
                                                                   "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            self._load_trials()
        previous_dataset = self.metadata.get("dataset")
        previous_protocol = self.metadata.get("proxy_protocol")
        self.metadata["device"] = self.device
        self.metadata["status"] = "running"
        self.metadata.pop("error", None)
        self.file = self.trial_path.open("a" if self.resume else "x", encoding="utf-8")
        try:
            if not self.smoke and not self.config.get("injected_evaluator"):
                # Fail on dataset/infrastructure errors before spending the search budget.
                from data.tabular import get_proxy_loaders, dataset_metadata
                current_dataset = dataset_metadata(self.config["split_seed"], self.config["dataset"])
                if self.resume and previous_dataset is not None and previous_dataset != current_dataset:
                    raise ValueError("Dataset provenance changed; refusing to resume")
                train, val = get_proxy_loaders(proxy_size=self.config["proxy_size"], seed=self.config["split_seed"],
                                               dataset=self.config["dataset"], validation_size=self.config["validation_size"])
                current_protocol = {"training_samples": len(train.dataset), "validation_samples": len(val.dataset),
                                                    "batch_size": train.batch_size,
                                                    "train_indices_sha256": hashlib.sha256(train.dataset.row_indices.tobytes()).hexdigest(),
                                                    "validation_indices_sha256": hashlib.sha256(val.dataset.row_indices.tobytes()).hexdigest()}
                if self.resume and previous_protocol is not None and previous_protocol != current_protocol:
                    raise ValueError("Proxy subsets changed; refusing to resume")
                self.metadata.update(dataset=current_dataset, proxy_protocol=current_protocol)
            self._write_metadata()
            print(f"Search metadata: {self.metadata_path}")
            if self.resume:
                print(f"Resuming {self.method}: reusing {len(self.replay)} completed trials")
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
            device = self.device
            args = (list(chromosome), device, self.config["proxy_epochs"], self._seed(trial_id),
                    self.config["split_seed"], self.config["proxy_size"], self.config.get("max_params"), self.config["dataset"], self.config["validation_size"])
            tasks.append((trial_id, args))
        fitnesses = []
        for trial_id, args in tasks:
            chromosome, device, epochs, seed, split_seed, proxy_size, max_params, dataset, validation_size = args
            if trial_id <= len(self.replay):
                record = self.replay[trial_id - 1]
                expected = dict(chromosome=chromosome, seed=seed, generation=generation,
                                split_seed=split_seed, proxy_size=proxy_size, dataset_name=dataset,
                                validation_size=validation_size, proxy_epochs=epochs, device=device)
                if any(record.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Search replay diverged at trial {trial_id}; refusing to resume")
                self._accept(record)
                fitnesses.append(record["fitness"])
                continue
            t0 = time.perf_counter()
            try:
                if self.smoke:
                    outcome = {"fitness": random.Random(seed).uniform(0.1, 0.9), "status": "smoke", "elapsed_s": 0.0}
                else:
                    outcome = self.evaluator(*args)
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
                      "dataset_name": dataset, "validation_size": validation_size, "proxy_epochs": epochs, "elapsed_s": outcome.get("elapsed_s", time.perf_counter() - t0)}
            self._record(record)
            fitnesses.append(fit)
        return fitnesses

    def _record(self, record):
        self.file.write(json.dumps(record, allow_nan=False) + "\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        if record["status"] != "error":
            self._accept(record)
        else:
            self.evaluation_seconds += record.get("elapsed_s", 0.0)
        self.metadata.update(evaluation_count=self.count,
                             elapsed_s=self.previous_elapsed + time.perf_counter() - self.started,
                             total_evaluation_seconds=self.evaluation_seconds)
        self._write_metadata()

    def _accept(self, record):
        self.count += 1
        self.evaluation_seconds += record.get("elapsed_s", 0.0)
        if record["status"] in ("ok", "smoke") and (self.best is None or record["fitness"] > self.best["fitness"]):
            self.best = record

    def finish(self, log_path=None):
        if self.best is None:
            raise RuntimeError("No valid candidate was evaluated; inspect trial failures")
        result = {"run_id": self.run_id, "method": self.method, "schema_version": SCHEMA_VERSION,
                  "dataset_name": self.config["dataset"], "search_space": SEARCH_SPACE, "smoke": self.smoke,
                  "winner_policy": "best_observed_validation_accuracy",
                  "best_chromosome": self.best["chromosome"], "best_arch": self.best["arch"],
                  "best_fitness": self.best["fitness"], "best_validation_loss": self.best.get("validation_loss"), "best_trial_id": self.best["trial_id"],
                  "hyperparams": self.config, "evaluation_count": self.count,
                  "elapsed_s": self.previous_elapsed + time.perf_counter() - self.started,
                  "total_evaluation_seconds": self.evaluation_seconds,
                  "trial_path": str(self.trial_path), "metadata_path": str(self.metadata_path),
                  "log_path": log_path}
        path = self.save_dir / f"best_{self.run_id}.json"
        result["save_path"] = str(path)
        atomic_json(path, result)
        return result

    def __exit__(self, kind, error, traceback):
        try:
            self.file.close()
            self.metadata.update(status="interrupted" if kind is KeyboardInterrupt else "failed" if error else "completed",
                                 evaluation_count=self.count,
                                 elapsed_s=self.previous_elapsed + time.perf_counter() - self.started,
                                 total_evaluation_seconds=self.evaluation_seconds)
            if error:
                self.metadata["error"] = str(error)
            self._write_metadata()
        finally:
            self.lock.release()


def resume_search(metadata_path, evaluator=None):
    """Replay deterministic controller decisions; train only unfinished trials."""
    path = Path(metadata_path).resolve()
    saved = json.loads(path.read_text(encoding="utf-8"))
    if saved.get("resume_version") != RESUME_VERSION:
        raise ValueError("This run predates resumable search; start a new run")
    config = saved["config"].copy()
    budget = config.pop("evaluation_budget")
    config.pop("injected_evaluator")
    for key in ("adaptive_mutation", "mutation_policy", "replacement_policy"):
        config.pop(key, None)
    common = dict(**config, smoke=saved["smoke"], evaluator=evaluator, resume=str(path),
                  log_dir=str(Path(saved["trial_path"]).parent), save_dir=saved["save_dir"])
    if saved["method"] == "ga":
        from ga.engine import run_nas
        return run_nas(evaluation_budget=budget, **common)
    if saved["method"] == "aging":
        from ga.aging import run_aging_evolution
        return run_aging_evolution(n_evaluations=budget, **common)
    if saved["method"] == "random":
        from models.baselines.random_nas import run_random_search
        return run_random_search(n_evaluations=budget, **common)
    raise ValueError("Unknown saved search method")
