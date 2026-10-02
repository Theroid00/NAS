"""
training/config.py
==================
Centralised hyperparameter constants for the NAS training pipeline.
Edit values here rather than scattering them across files.
"""

# ---- Proxy training (used during GA search) ----
PROXY_EPOCHS: int = 20           # Short training per architecture
PROXY_SUBSET_SIZE: int = 0      # All training rows for tabular proxy evaluation

# ---- Full training (best architecture only) ----
FULL_EPOCHS: int = 100
FULL_BATCH_SIZE: int = 32

# ---- Optimiser ----
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float  = 1e-4

# ---- Data ----
BATCH_SIZE: int = 32
NUM_WORKERS: int = 0            # DataLoader worker processes

# ---- CIFAR-10 normalisation (channel-wise mean/std) ----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---- Misc ----
NUM_CLASSES: int = 2
RANDOM_SEED: int = 42
