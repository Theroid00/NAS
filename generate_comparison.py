import matplotlib.pyplot as plt
import numpy as np
import os

def render_comparison():
    labels = ['ResNet Baseline', 'GA-NAS (Ours)', 'Random Search']
    accuracies = [90.52, 92.07, 92.15]
    colors = ['#888888', '#2ca02c', '#1f77b4']

    plt.figure(figsize=(8, 6))
    
    # Adjust ylim to emphasize the 90%+ differences
    plt.ylim(89.0, 93.0)
    plt.ylabel('Full Test Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('CIFAR-10 Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    
    bars = plt.bar(labels, accuracies, color=colors, width=0.5, edgecolor='black', linewidth=1.5)

    # Add text above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save the figure
    os.makedirs('experiments', exist_ok=True)
    save_path = 'experiments/final_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"Comparison plot successfully saved to {save_path}")

if __name__ == '__main__':
    render_comparison()
