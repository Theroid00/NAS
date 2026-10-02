"""
utils/logger.py
===============
Per-generation logging to CSV and console.
Creates run_<run_id>.csv in the configured tabular generation-log directory.
"""

import os
import csv
from ga.chromosome import chromosome_to_str


class GenerationLogger:
    """Logs GA generation or aging population stats to a CSV file."""

    COLUMNS = [
        "gen", "best_fitness", "mean_fitness", "std_fitness", "min_fitness",
        "best_chromosome", "elapsed_s",
    ]

    def __init__(self, log_dir: str, run_id: str, resume=False):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"run_{run_id}.csv")
        self._previous = {}
        if resume and os.path.exists(self.path):
            with open(self.path, newline="") as stream:
                self._previous = {row["gen"]: row for row in csv.DictReader(stream)}
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.COLUMNS)
        self._writer.writeheader()
        self._file.flush()

    def log(
        self,
        gen: int,
        best_fitness: float,
        mean_fitness: float,
        std_fitness: float,
        min_fitness: float,
        best_chromosome: list,
        elapsed: float,
    ):
        row = {
            "gen": gen,
            "best_fitness": round(best_fitness, 6),
            "mean_fitness": round(mean_fitness, 6),
            "std_fitness":  round(std_fitness, 6),
            "min_fitness":  round(min_fitness, 6),
            "best_chromosome": chromosome_to_str(best_chromosome),
            "elapsed_s": round(elapsed, 2),
        }
        previous = self._previous.get(str(gen))
        if previous and all(str(row[key]) == previous[key] for key in self.COLUMNS if key != "elapsed_s"):
            row["elapsed_s"] = previous["elapsed_s"]
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> str:
        """Close the file and return the path to the log."""
        self._file.close()
        return self.path
