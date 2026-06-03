# NAS with Genetic Algorithm

> CSE_5022 — Advanced Machine Learning | Problem Statement 10

A full Neural Architecture Search (NAS) system that uses a **Genetic Algorithm (GA)** to automatically discover optimal CNN architectures for image classification on **CIFAR-10**.

---

## Project Structure

```
nas-genetic/
├── ga/                     # Genetic Algorithm engine
│   ├── chromosome.py       # Search space & encoding / decoding
│   ├── operators.py        # Selection, crossover, mutation
│   ├── population.py       # Population helpers
│   └── engine.py           # Main GA loop
│
├── models/
│   ├── builder.py          # Dynamic CNN from chromosome
│   └── baselines/
│       ├── resnet.py       # Hand-designed ResNet baseline
│       └── random_nas.py   # Random search baseline
│
├── training/
│   ├── config.py           # Hyperparameter constants
│   ├── evaluator.py        # Proxy training + fitness
│   └── trainer.py          # Full training for best architecture
│
├── data/
│   └── cifar.py            # CIFAR-10 loaders (proxy + full)
│
├── utils/
│   ├── logger.py           # CSV generation logger
│   └── visualiser.py       # Convergence plots, architecture diagrams
│
├── experiments/            # Auto-created output directory
│   ├── generation_logs/    # run_<timestamp>.csv per GA run
│   └── best_architectures/ # best_<timestamp>.json + checkpoints
│
├── main.py                 # CLI entry point
├── evaluate_best.py        # Full-train the best found architecture
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Smoke test (CPU, no real training — verify everything works)
```bash
python main.py --smoke --pop 5 --gen 2
```

### 3. Full NAS run (single GPU)
```bash
python main.py --pop 20 --gen 15 --proxy-epochs 5 --device cuda
```

### 4. Dual-GPU parallel evaluation
```bash
python main.py --pop 20 --gen 15 --proxy-epochs 5 --device cuda --parallel
```

### 5. Full-train the best discovered architecture
```bash
python main.py --mode train-best --device cuda --full-epochs 50
# or equivalently:
python evaluate_best.py --device cuda --epochs 50
```

### 6. Random search baseline (same compute budget)
```bash
python main.py --mode random-search --n-eval 300 --device cuda
```

### 7. Plot convergence from a previous run
```bash
python main.py --mode plot
```

---

## Search Space (13 Genes)

| Gene | Name | Values |
|------|------|--------|
| 0 | `num_blocks` | 2, 3, 4, 5 |
| 1 | `filters_1` | 32, 64, 128 |
| 2 | `filters_2` | 64, 128, 256 |
| 3 | `filters_3` | 128, 256, 512 |
| 4 | `filters_4` | 256, 512, 1024 |
| 5 | `filters_5` | 512, 1024, 2048 |
| 6 | `kernel_size` | 3, 5 |
| 7 | `activation` | relu, leaky_relu, elu |
| 8 | `dropout` | 0.0, 0.2, 0.4, 0.5 |
| 9 | `batch_norm` | True, False |
| 10 | `fc_hidden` | 128, 256, 512 |
| 11 | `pooling` | max, avg, mixed |
| 12 | `use_residual` | True, False |

---

## Results & Baseline Comparison

We evaluated the discovered Genetic Algorithm architecture against a manual ResNet baseline and a compute-matched Random Search baseline (300 evaluations):

| Method / Baseline | Proxy Validation Accuracy (5 Epochs) | Full Test Accuracy (50 Epochs) | Trainable Parameters | Search Strategy / Budget |
| :--- | :---: | :---: | :---: | :--- |
| **GA-NAS (Ours)** | **72.15%** | **92.07%** | **19,061,642 (~19.06M)** | Evolutionary (15 Gen × 20 Pop = 300 evals) |
| **Random Search** | 69.90% | **92.15%** | 5,605,130 (~5.60M) | Random (300 random evaluations) |
| **Hand-designed ResNet** | — | 90.52% | 2,789,962 (~2.79M) | Human Baseline (3-stage ResNet) |

### Key Scientific Findings:
1. **The Proxy Gap:** During the architecture search phase, the GA successfully prioritized the best model candidate, achieving a proxy fitness of **72.15%** vs Random Search's **69.90%**. However, when trained to full convergence (50 epochs on the complete dataset), the Random Search winner achieved **92.15%** (a tiny 0.08% margin over the GA-NAS model). This demonstrates the **Proxy Gap** phenomenon in NAS—short, low-epoch training serves as a noisy fitness estimator, meaning initial optimization trajectories do not always correlate perfectly with final asymptotic performance.
2. **Discovered Design Choices:** The GA autonomously converged on a 5-block residual architecture (`use_residual=True`), utilizing LeakyReLU activation and Batch Normalization. Crucially, the GA chose to scale filter capacities up to **1024 channels** in the 4th block to prevent bottlenecking before pooling back down to 512 channels, showcasing automatic adaptation of channel capacity.

---

## GA Hyperparameters (defaults)

| Param | Value |
|-------|-------|
| Population size | 20 |
| Generations | 15 |
| Selection | Tournament (k=5) |
| Crossover | Single-point (p=0.8) |
| Mutation | Adaptive Per-gene (p=0.1, scales to 0.3 on stagnation) |
| Elitism | Top-2 carry forward |
| Proxy training | 10k subset, 5 epochs |

---

## Outputs

After a NAS run:

```
experiments/
├── ablations/        ← Generated by run_ablations.py comparison
├── sweep/            ← Generated by run_hyper_sweep.py heatmap
├── generation_logs/  ← per-gen stats (CSV)
└── best_architectures/
    ├── best_<timestamp>.json  ← best chromosome + fitness
    └── run_<timestamp>_convergence.png
```

After full training:

```
experiments/best_architectures/
├── ckpt_<run_id>_best.pt      ← PyTorch checkpoint
└── full_train_<run_id>.json   ← test accuracy + training history
```

---

## References

- Pham et al. (2018) *Efficient Neural Architecture Search via Parameter Sharing* (ENAS)
- Liu et al. (2019) *DARTS: Differentiable Architecture Search*
- Fortin et al. (2012) *DEAP: Evolutionary Algorithms Made Easy*
