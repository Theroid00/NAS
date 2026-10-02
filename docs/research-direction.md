# Saved research direction

Saved on 2026-10-02. This is parked; the current goal is an industry hiring portfolio.


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
