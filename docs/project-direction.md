# Project direction

Updated: 2026-10-02.

## Current goal: an industry AI/ML engineering portfolio

This project supports applications for software engineering roles, especially
AI/ML roles. It is not currently intended as a research paper. Prioritize a
usable, reproducible system and credible practical results over research novelty
or an extensive publication benchmark.

The story should demonstrate how to build, evaluate, debug, and operate an
automated model-search pipeline under a realistic single-GPU constraint.
Random search winning or tying is an acceptable result; the portfolio should
explain the tradeoffs honestly rather than guarantee that evolution wins.

### Proposed engineering priorities

1. Make the workflow straightforward: prepare data, run a configurable search,
   compare methods, retrain a winner, and evaluate an untouched test set.
2. Add useful classification reporting: balanced accuracy, macro F1, per-class
   recall, confusion matrices, and the existing accuracy and cross-entropy.
   Report runtime and model size alongside predictive performance.
3. Improve operational behavior where needed: resumable runs, clear failures,
   saved configuration, reproducible seeds, and discoverable artifacts.
4. Produce a bounded GPU comparison and a clear results report or demo that
   shows improvement over time and explains accuracy/compute tradeoffs.
5. Package the selected model for inference with documented preprocessing and
   an example interface. Verify that inference matches the saved model.
6. Present the project clearly: quick start, architecture overview, sample
   results, automated checks, limitations, and a short demo.

Covertype is adequate for the initial industry demonstration. Another dataset
is optional and should add a useful engineering capability, rather than just
more rows or a hoped-for advantage over random search. HIGGS would be a possible
later test of scalable loading and larger numeric classification workloads.

### Current constraints and decisions

- One RTX 4060 laptop GPU; sequential candidate training.
- Compare GA, aging evolution, and random search. Exclude the fixed MLP from
  the planned comparison suite; this does not request deletion of its code.
- Retain validation accuracy as the current search objective. Additional
  metrics are reporting metrics unless an objective change is explicitly chosen.
- Keep train/validation/test separation and honest, independently checked
  scoring even though publication-level experiments are not required.
- Continue committing approved work on `codex/reproducible-evolution-nas`.
- The large ten-seed publication suite is a previous proposal, not a committed
  requirement for the industry direction.
- This document saves the direction only. Metric implementation, new dataset
  downloads, and benchmark launches still require approval under the user's
  latest instruction to present the plan before acting.

## Parked research direction

Return to this section only if research publication becomes a goal again.

Potential question: when does evolution outperform random search on large
tabular tasks after controlling training noise, evaluation fidelity, and GPU
time? A possible extension would allocate extra training or repeated seeds to
uncertain, promising candidates. This overlaps existing work; novelty is not
established.

Relevant work found during the earlier literature review:

- [pTNAS, ICML 2026](https://proceedings.mlr.press/v306/xing26a.html): tabular
  architecture search with cheap filtering, staged training, and budget allocation.
- [R-MF-NAS, June 2026](https://doi.org/10.1145/3774940): proxy-guided search,
  successive halving, and robustness to proxy selection.
- [EvoDiff-NAS, August 2026](https://www.sciencedirect.com/science/article/pii/S2210650226001628):
  diffusion-guided mutations studied using precomputed NAS benchmarks.
- [AgEBO-Tabular](https://arxiv.org/abs/2010.16358): existing aging evolution
  combined with hyperparameter optimization for tabular data.
- [TabArena, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html):
  curated datasets and reproducible tabular model benchmarking.

Candidate datasets:

- [HIGGS](https://archive.ics.uci.edu/dataset/280/higgs): 11 million rows,
  28 numeric features; preserve the documented final 500,000-row test set.
- [HEPMASS](https://archive.ics.uci.edu/dataset/347/hepmass): 10.5 million rows
  per variant; related physics tasks, with limited domain diversity versus HIGGS.
- [Poker Hand](https://archive.ics.uci.edu/dataset/158/poker): 1,025,010 rows,
  10 card attributes; needs suitable categorical encoding and rare-class handling.

Dataset size alone does not imply that random search will perform poorly.
Choose datasets before observing comparative results.

The earlier publication proposal was proxy calibration on 12 architectures
with two training seeds, then three search methods across ten search seeds and
100 evaluations each, followed by three retraining seeds per winner. It also
included confidence intervals, multiple datasets, and a random-search plus
successive-halving control if staged resource allocation were introduced.
These are parked research ideas, not the current implementation plan.
