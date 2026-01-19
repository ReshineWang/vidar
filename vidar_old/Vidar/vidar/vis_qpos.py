import torch
import matplotlib.pyplot as plt
import os

file_path = "/data/dex/vidar/data/cube-human-test/fold_janes_twice_using_both_arms/episode_2_qpos.pt"
save_dir = "output"
save_name = "episode_0_qpos_vis.png"
save_path = os.path.join(save_dir, save_name)

# Ensure output directory exists
os.makedirs(save_dir, exist_ok=True)

try:
    # Load data
    data = torch.load(file_path, map_location='cpu')
    print(f"Loaded data shape: {data.shape}")

    # Check shape
    if len(data.shape) != 2 or data.shape[1] != 14:
        print(f"Warning: Expected shape [N, 14], got {data.shape}. Plotting anyway based on second dimension.")
    
    num_steps = data.shape[0]
    num_dims = data.shape[1]

    # Create figure with subplots
    # Just to be safe, calculation for grid
    import math
    cols = 4
    rows = math.ceil(num_dims / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()

    for i in range(num_dims):
        ax = axes[i]
        # data[:, i] contains the i-th dimension across all time steps
        ax.plot(range(num_steps), data[:, i].numpy(), color='blue', linewidth=1)
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")

except Exception as e:
    print(f"Error processing file: {e}")
