import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ga.aging import run_aging_evolution
from ga.engine import run_nas
from models.baselines.random_nas import run_random_search
from utils.persistence import RunLock
from utils.records import read_record
from utils.search_runtime import resume_search


def score(*args):
    # The search's RNG must also be independent of side effects in evaluation.
    random.seed(args[3])
    return {"fitness": random.random(), "status": "ok", "elapsed_s": 0.25}


class ResumeTests(unittest.TestCase):
    def launch(self, method, directory, evaluator=score, smoke=False):
        common = dict(seed=7, proxy_epochs=1, log_dir=directory, save_dir=directory,
                      evaluator=None if smoke else evaluator, smoke=smoke)
        if method == "ga":
            return run_nas(population_size=3, n_elites=1, evaluation_budget=23, **common)
        if method == "aging":
            return run_aging_evolution(population_size=3, n_evaluations=23, **common)
        return run_random_search(n_evaluations=23, **common)

    def records(self, path):
        return [json.loads(line) for line in Path(path).read_text().splitlines()]

    def test_resume_matches_uninterrupted_search_and_never_retrains_completed_trials(self):
        for method in ("ga", "aging", "random"):
            for stop in (1, 4, 21):
                with self.subTest(method=method, stop=stop), tempfile.TemporaryDirectory() as root:
                    uninterrupted = self.launch(method, str(Path(root) / "control"))
                    directory = Path(root) / "interrupted"
                    calls = []
                    def interrupt(*args):
                        if len(calls) == stop:
                            raise KeyboardInterrupt()
                        calls.append(args[3])
                        return score(*args)
                    with self.assertRaises(KeyboardInterrupt):
                        self.launch(method, str(directory), interrupt)
                    metadata_path = next(directory.glob("metadata_*.json"))
                    metadata = read_record(metadata_path)
                    self.assertEqual(metadata["status"], "interrupted")
                    self.assertEqual(metadata["evaluation_count"], stop)
                    completed_prefix = Path(metadata["trial_path"]).read_bytes()
                    resumed_calls = []
                    def remaining(*args):
                        resumed_calls.append(args[3])
                        return score(*args)
                    resumed = resume_search(metadata_path, remaining)
                    self.assertEqual(len(resumed_calls), 23 - stop)
                    self.assertTrue(Path(resumed["trial_path"]).read_bytes().startswith(completed_prefix))
                    original = self.records(uninterrupted["trial_path"])
                    recovered = [r for r in self.records(resumed["trial_path"]) if r["status"] != "error"]
                    self.assertEqual(original, recovered)
                    for key in ("best_chromosome", "best_trial_id", "best_fitness", "evaluation_count"):
                        self.assertEqual(uninterrupted[key], resumed[key])
                    self.assertAlmostEqual(resumed["total_evaluation_seconds"],
                                           sum(r["elapsed_s"] for r in self.records(resumed["trial_path"])))
                    # Resuming a completed run is idempotent and needs no evaluations.
                    replayed = resume_search(metadata_path, lambda *args: self.fail("Retrained completed run"))
                    self.assertEqual(replayed["run_id"], resumed["run_id"])

    def test_partial_write_recovery_and_repeated_interruptions(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def interrupt(*args):
                if len(calls) == 2:
                    raise RuntimeError("disconnected")
                calls.append(args[3])
                return score(*args)
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                self.launch("random", directory, interrupt)
            metadata = next(Path(directory).glob("metadata_*.json"))
            trial = Path(read_record(metadata)["trial_path"])
            with trial.open("ab") as stream:
                stream.write(b'{"trial_id": 3,')
            with self.assertRaisesRegex(RuntimeError, "again"):
                resume_search(metadata, lambda *args: (_ for _ in ()).throw(RuntimeError("again")))
            self.assertEqual(len(list(Path(directory).glob("*.partial.*"))), 1)
            result = resume_search(metadata, score)
            self.assertEqual(result["evaluation_count"], 23)
            self.assertEqual(len([r for r in self.records(trial) if r["status"] == "error"]), 2)

    def test_replay_divergence_and_corruption_are_rejected(self):
        for corruption in ("chromosome", "journal"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                result = self.launch("random", directory)
                path = Path(result["trial_path"])
                records = self.records(path)
                if corruption == "chromosome":
                    from ga.chromosome import decode
                    records[0]["chromosome"][0] = (records[0]["chromosome"][0] + 1) % 4
                    records[0]["arch"] = decode(records[0]["chromosome"])
                    path.write_text("".join(json.dumps(record) + "\n" for record in records))
                    message = "diverged"
                else:
                    path.write_text('{invalid}\n')
                    message = "corrupt"
                with self.assertRaisesRegex(ValueError, message):
                    resume_search(result["metadata_path"], score)

    def test_active_run_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.launch("random", directory)
            lock = RunLock(Path(result["metadata_path"]).with_suffix(".lock"))
            lock.acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    resume_search(result["metadata_path"], score)
            finally:
                lock.release()

    def test_legacy_format_and_configuration_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.launch("random", directory)
            path = Path(result["metadata_path"])
            metadata = json.loads(path.read_text())
            with self.assertRaisesRegex(ValueError, "configuration"):
                run_random_search(n_evaluations=23, seed=8, proxy_epochs=1, log_dir=directory,
                                  save_dir=directory, evaluator=score, resume=path)
            metadata.pop("resume_version")
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "predates"):
                resume_search(path, score)

    def test_comparison_resumes_partial_search_and_skips_completed_methods(self):
        from run_comparison import compare, resume_comparison
        from utils.search_runtime import SearchSession
        original = SearchSession.evaluate
        with tempfile.TemporaryDirectory() as directory:
            control = compare(["ga", "aging", "random"], [1, 2], budget=7, population=3,
                              smoke=True, out_dir=directory)
            def interrupted(session, population, generation=None):
                if session.method == "random":
                    original(session, population[:2], generation)
                    raise KeyboardInterrupt()
                return original(session, population, generation)
            with patch.object(SearchSession, "evaluate", interrupted), self.assertRaises(KeyboardInterrupt):
                compare(["ga", "aging", "random"], [1, 2], budget=7, population=3,
                        smoke=True, out_dir=directory)
            path = next(p for p in Path(directory).glob("*/comparison.json")
                        if json.loads(p.read_text())["status"] == "interrupted")
            prefix = read_record(path)["runs"]
            recovered = resume_comparison(path)
            self.assertEqual(recovered["runs"][:2], prefix)
            self.assertEqual([(r["method"], r["seed"], r["best_proxy_fitness"]) for r in recovered["runs"]],
                             [(r["method"], r["seed"], r["best_proxy_fitness"]) for r in control["runs"]])
            with patch("run_comparison.run_nas", side_effect=AssertionError("Reran GA")), patch(
                    "run_comparison.run_random_search", side_effect=AssertionError("Reran random")), patch(
                    "run_comparison.run_aging_evolution", side_effect=AssertionError("Reran aging")):
                self.assertEqual(resume_comparison(path)["runs"], recovered["runs"])

    def test_completed_final_training_is_recovered_after_manifest_write_gap(self):
        from run_comparison import compare, resume_comparison
        with tempfile.TemporaryDirectory() as directory:
            completed = compare(["random"], [1], budget=3, population=2, proxy_epochs=1,
                                full_epochs=1, full_seeds=[101], out_dir=directory)
            path = next(Path(directory).glob("*/comparison.json"))
            recorded_full = completed["runs"][0]["full_training"][:]
            completed["runs"][0]["full_training"] = []
            path.write_text(json.dumps(completed))
            with patch("training.trainer.full_train", side_effect=AssertionError("Retrained completed winner")):
                recovered = resume_comparison(path)
            self.assertEqual(recovered["runs"][0]["full_training"], recorded_full)

    def test_gpu_training_resume_matches_uninterrupted_run(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        from utils.search_runtime import evaluate_task
        with tempfile.TemporaryDirectory() as directory:
            common = dict(n_evaluations=4, population_size=2, proxy_epochs=2, device="cuda",
                          log_dir=directory, save_dir=directory, seed=17)
            control = run_aging_evolution(**common)
            calls = []
            def interrupt(*args):
                if len(calls) == 2:
                    raise KeyboardInterrupt()
                calls.append(args[3])
                return evaluate_task(*args)
            with patch("utils.search_runtime.evaluate_task", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
                run_aging_evolution(**common)
            metadata = next(p for p in Path(directory).glob("metadata_*.json")
                            if json.loads(p.read_text())["status"] == "interrupted")
            with patch("utils.search_runtime.evaluate_task", wraps=evaluate_task) as evaluator:
                resumed = resume_search(metadata)
            self.assertEqual(evaluator.call_count, 2)
            original = self.records(control["trial_path"])
            recovered = [r for r in self.records(resumed["trial_path"]) if r["status"] == "ok"]
            self.assertEqual([(r["chromosome"], r["fitness"], r["validation_loss"]) for r in original],
                             [(r["chromosome"], r["fitness"], r["validation_loss"]) for r in recovered])

    def test_process_death_releases_lock_and_keeps_completed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            script = '''
import os, random, sys
from models.baselines.random_nas import run_random_search
calls = 0
def evaluator(*args):
    global calls
    calls += 1
    if calls == 4:
        os._exit(73)
    random.seed(args[3])
    return {"fitness": random.random(), "status": "ok", "elapsed_s": 0.25}
run_random_search(n_evaluations=23, seed=7, proxy_epochs=1,
                  log_dir=sys.argv[1], save_dir=sys.argv[1], evaluator=evaluator)
'''
            process = subprocess.run([sys.executable, "-c", script, directory],
                                     cwd=Path(__file__).resolve().parents[1], capture_output=True, timeout=30)
            self.assertEqual(process.returncode, 73, process.stderr.decode())
            metadata = next(Path(directory).glob("metadata_*.json"))
            saved = read_record(metadata)
            self.assertEqual(saved["status"], "running")
            self.assertEqual(saved["evaluation_count"], 3)
            calls = []
            def remaining(*args):
                calls.append(args[3])
                return score(*args)
            recovered = resume_search(metadata, remaining)
            self.assertEqual(len(calls), 20)
            self.assertEqual(recovered["evaluation_count"], 23)

    def test_changed_dataset_is_rejected_without_replacing_original_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_random_search(n_evaluations=2, proxy_epochs=1, log_dir=directory, save_dir=directory)
            path = Path(result["metadata_path"])
            original = json.loads(path.read_text())["dataset"]
            with patch("data.tabular.dataset_metadata", return_value={**original, "data_sha256": "changed"}):
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, "provenance changed"):
                        resume_search(path)
                    self.assertEqual(json.loads(path.read_text())["dataset"], original)

    def test_complete_final_record_without_newline_can_be_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.launch("random", directory)
            path = Path(result["trial_path"])
            path.write_bytes(path.read_bytes().rstrip(b"\r\n"))
            replayed = resume_search(result["metadata_path"], lambda *args: self.fail("Retrained"))
            self.assertEqual(replayed["evaluation_count"], 23)
            self.assertTrue(path.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
