import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ga.engine import run_nas

def main():
    pop_sizes = [10, 20, 30]
    epoch_sizes = [3, 5, 8]
    
    results = np.zeros((len(pop_sizes), len(epoch_sizes)))
    
    print("Running Hyperparameter Sweep...")
    for i, pop in enumerate(pop_sizes):
        for j, ep in enumerate(epoch_sizes):
            print(f"\n--- Sweeping Pop={pop}, Epochs={ep} ---")
            res = run_nas(
                population_size=pop,
                n_generations=5,  # short run for sweeping
                proxy_epochs=ep,
                device="cuda",
                log_dir=f"experiments/sweep",
                save_dir=f"experiments/sweep",
            )
            results[i, j] = res["best_fitness"]
            
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        results, 
        annot=True, 
        fmt=".4f", 
        xticklabels=epoch_sizes, 
        yticklabels=pop_sizes,
        cmap="viridis"
    )
    ax.set_xlabel("Proxy Epochs")
    ax.set_ylabel("Population Size")
    ax.set_title("Hyperparameter Sensitivity (Best Fitness at Gen 5)")
    
    out_path = "experiments/sweep/hyperparam_heatmap.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"\nSweep heatmap saved to {out_path}")

if __name__ == "__main__":
    main() 
