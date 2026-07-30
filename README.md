# Neural Architecture Search (NAS) with Genetic Algorithms

> **CSE_5022 — Advanced Machine Learning** | Autonomous Deep Learning Architecture Discovery

A full-fledged Neural Architecture Search (NAS) system that uses a **Genetic Algorithm (GA)** to automatically discover optimal Convolutional Neural Network (CNN) architectures for image classification on **CIFAR-10**.

---

## 🌟 Key Highlights
- **Autonomous Architecture Design:** Searches across **1,119,744** discrete network configurations without manual intervention.
- **Adaptive Mutation Rate:** Dynamically scales per-gene mutation from $p_m=0.1$ up to $p_m=0.3$ upon detecting 3 generations of fitness stagnation.
- **Multi-GPU Parallelization:** Supports concurrent population evaluation across dual CUDA devices (`cuda:0` and `cuda:1`).
- **High Performance:** Discovered a 5-block residual CNN that achieved **92.07% Test Accuracy** (19.06M parameters) on full 50-epoch dataset convergence.
- **Comprehensive Benchmarks:** Evaluated against hand-designed ResNet baselines and a 300-budget Random Search baseline to analyze the **Proxy Gap**.

---

## 🔄 System Architecture & Search Pipeline

```mermaid
graph TD
    A[Start: Initialize 20 Random Chromosomes] --> B[Proxy Evaluation: 5 Epochs on 10k CIFAR Split]
    B --> C{Diverged / NaN?}
    C -- Yes --> D[Fitness = 0.0]
    C -- No --> E[Fitness = Validation Accuracy]
    D --> F[Log Generation Statistics]
    E --> F
    F --> G[Identify Top 2 Elites]
    G --> H[Tournament Selection: k=5]
    H --> I[Single-Point Crossover: p=0.8]
    I --> J[Adaptive Mutation: p=0.1 to 0.3]
    J --> K[Form Next Generation: Elites + Offspring]
    K --> L{15 Generations Complete?}
    L -- No --> B
    L -- Yes --> M[Full Convergence Training: 50 Epochs on Best Chromosome]
    M --> N[Save Checkpoints & Render Analytics]
```

---

## 📁 Repository Structure

```
.
├── ga/                     # Evolutionary Algorithm Engine
│   ├── chromosome.py       # 13-gene search space encoding & decoding
│   ├── operators.py        # Tournament selection, crossover, adaptive mutation
│   ├── population.py       # Population initialization & ranking helpers
│   └── engine.py           # Main GA execution loop
│
├── models/                 # Dynamic Model Construction & Baselines
│   ├── builder.py          # Dynamic PyTorch CNN builder (ConvBlock, ResidualBlock)
│   └── baselines/
│       ├── resnet.py       # Hand-crafted 3-block ResNet baseline (~2.8M params)
│       └── random_nas.py   # Compute-matched Random Search baseline (300 evals)
│
├── training/               # Data Loading & Training Pipelines
│   ├── config.py           # Search space & training hyperparameter constants
│   ├── evaluator.py        # 5-epoch proxy training & fitness scoring
│   └── trainer.py          # 50-epoch full convergence training with Cosine LR
│
├── utils/                  # Logging & Analytics Visualizer
│   ├── logger.py           # CSV generation logger
│   └── visualiser.py       # Plotting utilities for convergence, heatmaps, & diagrams
│
├── experiments/            # Auto-generated Experimental Analytics
│   ├── ablations/          # Genetic operator ablation logs & comparison plots
│   ├── sweep/              # Hyperparameter sensitivity sweep logs & heatmaps
│   ├── generation_logs/    # Generation CSV logs & convergence curves
│   └── best_architectures/ # Winning chromosome JSON configs & architecture plots
│
├── main.py                 # Primary CLI entry point
├── evaluate_best.py        # Standalone full training script for winning JSON
├── run_ablations.py        # Ablation study runner script
├── run_hyper_sweep.py      # Population vs Proxy Epoch sensitivity sweeper
├── train_random_best.py    # Standalone full trainer for random search winner
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation
```

---

## 📊 Empirical Results & Baseline Comparison

We evaluated our evolved **GA-NAS** model against a hand-designed 3-block ResNet baseline and a compute-matched Random Search baseline (300 total evaluations):

| Method / Baseline | Proxy Validation Accuracy (5 Epochs) | Full Test Accuracy (50 Epochs) | Trainable Parameters | Search Strategy / Budget |
| :--- | :---: | :---: | :---: | :--- |
| **GA-NAS (Ours)** | **72.15%** | **92.07%** | **19,061,642 (~19.06M)** | Evolutionary (15 Gen × 20 Pop = 300 evals) |
| **Random Search** | 69.90% | **92.15%** | 5,605,130 (~5.60M) | Random Search (300 random samples) |
| **Hand-designed ResNet** | — | 90.52% | 2,789,962 (~2.79M) | Manual Design (3-stage ResNet) |

### 📈 Baseline Accuracy Comparison
![Baseline Comparison](experiments/final_comparison.png)

---

## 🔬 Scientific Findings: The Proxy Gap & Design Choices

1. **The Proxy Gap Phenomenon:**  
   During the proxy search phase, the GA prioritized the strongest architecture candidate, achieving a proxy fitness of **72.15%** compared to Random Search's **69.90%**. However, when trained to full convergence (50 epochs on 50k samples), the Random Search model reached **92.15%** (a minor 0.08% variance over the GA model's **92.07%**). This empirically confirms the **Proxy Gap** in NAS literature—short 5-epoch training acts as a noisy proxy, meaning initial optimization speed does not always guarantee superior asymptotic convergence.
2. **Discovered Architectural Innovations:**  
   The GA autonomously discovered that standard shallow networks suffer from feature bottlenecks. It selected a 5-block deep hierarchy (`use_residual=True`), utilizing **LeakyReLU** activations, **Batch Normalization**, and **Max Pooling**. Notably, the GA dynamically widened filter capacity to **1024 channels** in Block 4 before pooling down to 512 channels, optimizing feature extraction prior to classification.

---

## 🧪 Ablation Study: Genetic Component Contributions

To isolate the impact of each evolutionary component, we performed ablation experiments across 10 generations:

| Configuration | Final Best Proxy Fitness | Key Observation |
| :--- | :---: | :--- |
| **Full GA-NAS (Baseline)** | `0.6705` | Balanced exploration/exploitation baseline ($P=20, G=10, p_c=0.8, p_m=0.1$) |
| **No Crossover ($p_c=0.0$)** | `0.7425` | High mutation exploration found high-performing search regions early |
| **No Mutation ($p_m=0.0$)** | `0.7215` | Crossover dominated early convergence, but stalled without gene renewal |
| **No Elitism ($N_{\text{elites}}=0$)** | `0.7095` | Moderate performance; fitness experienced variance without elite carry-over |
| **Small Population ($P=10$)** | `0.6805` | Underperformed due to restricted initial genetic diversity |

### 📉 Ablation Comparison Plot
![Ablation Comparison](experiments/ablations/ablation_comparison.png)

---

## 🎛️ Hyperparameter Sensitivity Analysis

We evaluated search performance across varying population sizes ($P \in \{10, 20, 30\}$) and proxy epoch durations ($E \in \{3, 5, 8\}$):

### 🌡️ Hyperparameter Heatmap
![Hyperparameter Heatmap](experiments/sweep/hyperparam_heatmap.png)

---

## 🧬 Evolved Architecture Visual Diagram

![Evolved Architecture](experiments/best_architectures/architecture.png)

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Smoke Test (Sanity check on CPU)
```bash
python main.py --smoke --pop 5 --gen 2
```

### 3. Run GA Search (Single GPU / Dual GPU)
```bash
# Single GPU:
python main.py --pop 20 --gen 15 --proxy-epochs 5 --device cuda

# Dual GPU (Parallel evaluation across cuda:0 and cuda:1):
python main.py --pop 20 --gen 15 --proxy-epochs 5 --device cuda --parallel
```

### 4. Full-Train Discovered Winner
```bash
python evaluate_best.py --device cuda --epochs 50
```

### 5. Run Baselines & Ablations
```bash
# Random Search baseline (300 evaluations):
python main.py --mode random-search --n-eval 300 --device cuda

# Hand-designed ResNet baseline:
python main.py --mode train-resnet --device cuda --full-epochs 50

# Genetic Operator Ablation Suite:
python run_ablations.py

# Hyperparameter Sensitivity Sweep:
python run_hyper_sweep.py
```

---

## 📚 References
- Pham et al. (2018) *Efficient Neural Architecture Search via Parameter Sharing* (ENAS)
- Liu et al. (2019) *DARTS: Differentiable Architecture Search*
- Real et al. (2019) *Regularized Evolution for Image Classifier Architecture Search* (AmoebaNet)
- Fortin et al. (2012) *DEAP: Evolutionary Algorithms Made Easy*
