"""
training/config.py
==================
Centralised hyperparameter constants for the NAS training pipeline.
Edit values here rather than scattering them across files.
"""

# ---- Proxy training (used during GA search) ----
PROXY_EPOCHS: int = 5           # Short training per architecture
PROXY_SUBSET_SIZE: int = 10_000 # Samples from CIFAR-10 for proxy eval

# ---- Full training (best architecture only) ----
FULL_EPOCHS: int = 50
FULL_BATCH_SIZE: int = 128

# ---- Optimiser ----
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float  = 1e-4

# ---- Data ----
BATCH_SIZE: int = 128
NUM_WORKERS: int = 2            # DataLoader worker processes

# ---- CIFAR-10 normalisation (channel-wise mean/std) ----
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# ---- Misc ----
NUM_CLASSES: int = 10
RANDOM_SEED: int = 42
