import argparse
import os
import copy
from ga.engine import run_nas
from utils.visualiser import plot_convergence

def run_ablation(name, **kwargs):
    print(f"\n{'='*40}\nRunning Ablation: {name}\n{'='*40}")
    os.environ["WANDB_DISABLED"] = "true"
    result = run_nas(
        population_size=kwargs.get("pop", 20),
        n_generations=10,
        proxy_epochs=5,
        crossover_prob=kwargs.get("cx", 0.8),
        mutation_prob=kwargs.get("mut", 0.1),
        tournament_k=kwargs.get("tk", 5),
        n_elites=kwargs.get("elites", 2),
        device="cuda",
        log_dir=f"experiments/ablations/{name}",
        save_dir=f"experiments/ablations/{name}"
    )
    return result["log_path"]

def main():
    os.makedirs("experiments/ablations", exist_ok=True)
    
    logs = {}
    
    # 1. Baseline GA (already has our new adaptive mutation and k=5)
    logs["baseline"] = run_ablation("baseline")
    
    # 2. No crossover
    logs["no_crossover"] = run_ablation("no_crossover", cx=0.0)
    
    # 3. No mutation
    logs["no_mutation"] = run_ablation("no_mutation", mut=0.0)
    
    # 4. No elitism
    logs["no_elitism"] = run_ablation("no_elitism", elites=0)
    
    # 5. Smaller population
    logs["small_pop"] = run_ablation("small_pop", pop=10)
    
    # Plotting them all together
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="darkgrid", context="paper")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, log_path in logs.items():
        df = pd.read_csv(log_path)
        ax.plot(df["gen"], df["best_fitness"], marker="o", label=name)
        
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Validation Accuracy (Proxy)")
    ax.set_title("Ablation Study: Component Contributions")
    ax.legend()
    
    out_path = "experiments/ablations/ablation_comparison.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"\nAblation plot saved to: {out_path}")

if __name__ == "__main__":
    main()
