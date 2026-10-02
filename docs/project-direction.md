# Project direction

Updated: 2026-10-03.

## Current goal: an industry AI/ML engineering portfolio

This project supports applications for software engineering roles, especially
AI/ML roles. It is not currently intended as a research paper. Prioritize a
usable, reproducible system and credible practical results over research novelty
or an extensive publication benchmark.

The story should demonstrate how to build, evaluate, debug, and operate an
automated model-search pipeline under a realistic single-GPU constraint.
Random search winning or tying is an acceptable result; the portfolio should
explain the tradeoffs honestly rather than guarantee that evolution wins.

### Engineering priorities

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
- Continue committing approved work on `resume/resume-trial`.
- The large ten-seed publication suite is a previous proposal, not a committed
  requirement for the industry direction.

### Implemented industry workflow

The user approved implementation on `resume/resume-trial`. Class-balanced metrics,
trial recovery, GPU comparison and finalization, portable model export, a prediction
API, and an interactive dashboard are implemented. The RTX 4060 suite completed
750 candidate evaluations and 15 winner retrainings; final test evaluation and
inference measurements also used CUDA. See [industry-demo.md](industry-demo.md)
and [industry-results.json](industry-results.json).

The research direction is preserved separately in
[research-direction.md](research-direction.md) for a future change of goal.
