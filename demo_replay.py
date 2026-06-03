import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

def replay_demo(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(df['gen'].min(), df['gen'].max())
    ax.set_ylim(max(0, df['min_fitness'].min() - 0.1), df['best_fitness'].max() + 0.1)
    
    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Live Demo: NAS Convergence")
    
    best_line, = ax.plot([], [], "o-", color="blue", label="Best Fitness", linewidth=2)
    mean_line, = ax.plot([], [], "s--", color="orange", label="Mean Fitness", linewidth=1.5)
    
    fill_collection = None
    ax.legend()
    
    def init():
        best_line.set_data([], [])
        mean_line.set_data([], [])
        return best_line, mean_line
        
    def animate(i):
        nonlocal fill_collection
        
        frame_df = df.iloc[:i+1]
        x = frame_df['gen'].values
        
        best_line.set_data(x, frame_df['best_fitness'].values)
        mean_line.set_data(x, frame_df['mean_fitness'].values)
        
        if fill_collection is not None:
            fill_collection.remove()
            
        fill_collection = ax.fill_between(
            x, 
            frame_df['min_fitness'].values, 
            frame_df['best_fitness'].values,
            alpha=0.2, color="blue"
        )
        
        return best_line, mean_line
        
    ani = animation.FuncAnimation(
        fig, animate, frames=len(df), init_func=init, 
        interval=500, blit=False, repeat=False
    )
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to generation log CSV")
    args = parser.parse_args()
    replay_demo(args.csv)
