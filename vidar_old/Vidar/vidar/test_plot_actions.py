
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from idm.idm import IDM
from idm.preprocessor import DinoPreprocessor
from idm.cache_dataset import CacheDataSet
from idm.utils import seed_torch
import cv2
from PIL import Image
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Test and Plot Actions")
    parser.add_argument("--load_from", type=str, required=True, help="Path to checkpoint")
    # Dataset path is optional if single video/action files are provided
    parser.add_argument("--dataset_path", type=str, default="", help="Test dataset path directory")
    parser.add_argument("--save_dir", type=str, default="output/plots", help="Directory to save plots")
    parser.add_argument("--model_name", type=str, default="mask", help="Model name")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (recommend 1 for plotting sequential)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--use_normalization", action="store_true", default=False)
    parser.add_argument("--use_transform", action="store_true", default=False) 
    parser.add_argument("--load_mp4", action="store_true", default=True)
    
    # New arguments for single file mode
    parser.add_argument("--video_path", type=str, default="", help="Path to single .mp4 video file")
    parser.add_argument("--action_path", type=str, default="", help="Path to single .pt action file")

    return parser.parse_args()

def plot_actions(gt_actions, pred_actions, save_dir, sample_idx):
    """
    gt_actions: [T, 14] numpy array
    pred_actions: [T, 14] numpy array
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_steps = gt_actions.shape[0]
    num_dims = gt_actions.shape[1]

    # Calculate grid layout
    cols = 4
    rows = math.ceil(num_dims / cols)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    time_steps = np.arange(num_steps)

    for i in range(num_dims):
        ax = axes[i]
        
        # Plot GT
        ax.plot(time_steps, gt_actions[:, i], color='blue', label='Ground Truth', linewidth=1)
        
        # Plot Pred (ensure dimension exists)
        if i < pred_actions.shape[1]:
            ax.plot(time_steps, pred_actions[:, i], color='red', label='Predicted', linestyle='--', linewidth=1)
            
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
    # Save the combined plot
    save_name = f"sample_{sample_idx}_comparison.png"
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved successfully to: {save_path}")

def process_single_video(args, net, preprocessor, accelerator):
    """
    Process a single video/action pair bypassing the CacheDataSet directory scan.
    """
    print(f"Processing single video mode.")
    print(f"Video: {args.video_path}")
    print(f"Action: {args.action_path}")
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    if not os.path.exists(args.action_path):
        print(f"Error: Action file not found: {args.action_path}")
        return

    # 1. Load Ground Truth Actions
    gt_data = torch.load(args.action_path, map_location='cpu', weights_only=False)
    
    # Handle different .pt formats (Tensor vs Dict)
    if isinstance(gt_data, dict):
        if 'action' in gt_data:
            gt_data = gt_data['action']
        elif 'qpos' in gt_data:
             gt_data = gt_data['qpos']
        else:
             for k, v in gt_data.items():
                 if isinstance(v, torch.Tensor):
                     gt_data = v
                     break
    
    if not isinstance(gt_data, torch.Tensor):
        print(f"Error: Could not extract action tensor from {args.action_path}")
        return

    gt_tensor = gt_data # [T, D]
    
    # 2. Iterate Video Frames & Inference
    cap = cv2.VideoCapture(args.video_path)
    episode_pred = []
    
    print("Starting inference on video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing: BGR -> RGB -> PIL -> Preprocessor -> Tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        input_tensor = preprocessor.process_image(pil_img)
        input_tensor = input_tensor.unsqueeze(0).to(accelerator.device)
        
        output = net(input_tensor, return_mask=False)
        
        if args.use_normalization:
            output = net.normalize(output)
            
        episode_pred.append(output.detach().cpu().numpy())

    cap.release()
    
    if not episode_pred:
        print("Error: No frames read from video.")
        return

    # 3. Concatenate and Align
    pred_arr = np.concatenate(episode_pred, axis=0) # [T_video, 14]
    gt_arr = gt_tensor.numpy()                      # [T_action, 14]
    
    # Truncate to the shorter length to align plots
    length = min(len(pred_arr), len(gt_arr))
    
    if length == 0:
        print("Error: Length mismatch resulted in 0 overlap.")
        return

    print(f"Video frames: {len(pred_arr)}, GT steps: {len(gt_arr)}. Plotting first {length} steps.")
    
    pred_arr = pred_arr[:length]
    gt_arr = gt_arr[:length]
    
    # 4. Plot
    plot_actions(gt_arr, pred_arr, args.save_dir, "single_custom")


def main():
    args = parse_args()
    seed_torch(1234)
    accelerator = Accelerator()
    
    # Init Model
    net = IDM(model_name=args.model_name, output_dim=14)
    if args.use_normalization:
        pass 

    # Load Checkpoint
    try:
        print(f"Loading checkpoint from {args.load_from}")
        loaded_dict = torch.load(args.load_from, map_location='cpu', weights_only=False)
        
        if "model_state_dict" in loaded_dict:
            state_dict = loaded_dict["model_state_dict"]
        else:
            state_dict = loaded_dict
            
        net.load_state_dict(state_dict)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    net.to(accelerator.device)
    net.eval()

    # Preprocessor
    preprocessor = DinoPreprocessor(args)
    
    # Branch: Single File vs Dataset
    if args.video_path and args.action_path:
        # Single File Mode
        with torch.no_grad():
            process_single_video(args, net, preprocessor, accelerator)
            
    elif args.dataset_path:
        # Dataset Directory Mode
        print(f"Loading dataset from {args.dataset_path}")
        dataset = CacheDataSet(args, dataset_path=args.dataset_path, disable_pbar=False, type="test", preprocessor=preprocessor)
        print(f"Dataset size: {len(dataset)}")

        total_videos = len(dataset.video_lengths)
        
        with torch.no_grad():
            for vid_idx in range(total_videos):
                print(f"Processing video {vid_idx+1}/{total_videos}")
                
                start = dataset.data_begin[vid_idx]
                length = dataset.video_lengths[vid_idx]
                
                episode_gt = []
                episode_pred = []
                
                for i in range(length):
                    global_idx = start + i
                    image, pos = dataset[global_idx] 
                    
                    image = image.unsqueeze(0).to(accelerator.device)
                    pos = pos.unsqueeze(0).to(accelerator.device)
                    
                    output = net(image, return_mask=False)
                    
                    if args.use_normalization:
                        output = net.normalize(output) 
                        pos = net.normalize(pos)

                    episode_gt.append(pos.cpu().numpy())
                    episode_pred.append(output.cpu().numpy())
                
                gt_arr = np.concatenate(episode_gt, axis=0) # [T, 14]
                pred_arr = np.concatenate(episode_pred, axis=0) # [T, 14]
                
                plot_actions(gt_arr, pred_arr, args.save_dir, vid_idx)
                
                if vid_idx >= 4: 
                    break

        print(f"Plots saved to {args.save_dir}")
    else:
        print("Error: You must provide either --dataset_path OR (--video_path AND --action_path)")

if __name__ == "__main__":
    main()
