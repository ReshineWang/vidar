import torch
import os
import glob
import argparse
from tqdm import tqdm

def calculate_stats(dataset_path):
    print(f"Scanning dataset at: {dataset_path}")
    
    # Search for all qpos/action .pt files
    # Heuristics: search for *_qpos.pt or *_action.pt
    files = glob.glob(os.path.join(dataset_path, "**", "*_qpos.pt"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(dataset_path, "**", "*_action.pt"), recursive=True)
    
    if not files:
        print("No .pt files found ending with _qpos.pt or _action.pt")
        return

    print(f"Found {len(files)} files. Calculating statistics...")

    all_data = []

    for f in tqdm(files):
        try:
            data = torch.load(f, map_location='cpu', weights_only=False)
            
            # Helper to extract tensor
            tensor_data = None
            if isinstance(data, torch.Tensor):
                tensor_data = data
            elif isinstance(data, dict):
                for key in ['action', 'qpos', 'joint_position']:
                    if key in data:
                        tensor_data = data[key]
                        break
                if tensor_data is None:
                    # Fallback: take first tensor value
                    for v in data.values():
                        if isinstance(v, torch.Tensor):
                            tensor_data = v
                            break
            
            if tensor_data is not None:
                # Ensure 2D [T, D]
                if tensor_data.dim() == 1:
                    tensor_data = tensor_data.unsqueeze(0)
                all_data.append(tensor_data)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    # Concatenate all data along time dimension [Total_Time, Dim]
    full_tensor = torch.cat(all_data, dim=0)
    
    print(f"Total samples: {full_tensor.shape[0]}")
    print(f"Dimension: {full_tensor.shape[1]}")

    mean = torch.mean(full_tensor, dim=0)
    std = torch.std(full_tensor, dim=0)

    # Avoid division by zero for constant dimensions
    std[std < 1e-6] = 1.0

    print("\n" + "="*50)
    print("COPY AND PASTE THE FOLLOWING INTO idm.py:")
    print("="*50)
    
    # Format for code paste
    mean_str = ", ".join([f"{x:.8f}" for x in mean.tolist()])
    std_str = ", ".join([f"{x:.8f}" for x in std.tolist()])
    
    print(f"train_mean = torch.tensor([{mean_str}])")
    print(f"train_std = torch.tensor([{std_str}])")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of your dataset")
    args = parser.parse_args()
    
    calculate_stats(args.dataset_path)
