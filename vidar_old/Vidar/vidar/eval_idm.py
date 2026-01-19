import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from accelerate import Accelerator
from idm.idm import IDM, Mask
from idm.preprocessor import DinoPreprocessor
from idm.cache_dataset import CacheDataSet
from idm.utils import seed_torch
import cv2
from PIL import Image
import math

def parse_args():
    parser = argparse.ArgumentParser(description="Eval IDM")
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
    parser.add_argument("--use_gt_mask", action="store_true", default=False, help="Use ground truth mask")
    parser.add_argument("--mask_weight", type=float, default=1e-3, help="Mask weight") # Add for compatibility if Preprocessor needs args
    parser.add_argument("--ratio_eval", type=float, default=0.05) # Add for compatibility
    
    # New arguments for single file mode
    parser.add_argument("--video_path", type=str, default="", help="Path to single .mp4 video file")
    parser.add_argument("--action_path", type=str, default="", help="Path to single .pt action file")
    parser.add_argument("--mask_path", type=str, default="", help="Path to single .mp4 mask file (optional unless use_gt_mask is True)")
    parser.add_argument("--hdf5_path", type=str, default="", help="Path to single .hdf5 file")

    return parser.parse_args()

import h5py
import io

def process_single_hdf5(args, net, preprocessor, accelerator):
    print(f"Processing single HDF5 mode: {args.hdf5_path}")
    
    if not os.path.exists(args.hdf5_path):
        print(f"Error: HDF5 file not found: {args.hdf5_path}")
        return

    with h5py.File(args.hdf5_path, 'r') as f:
        # 1. Load Actions
        # Assuming RoboTwin structure: joint_action/vector
        if 'joint_action' in f and 'vector' in f['joint_action']:
            gt_actions = f['joint_action']['vector'][:] # [T, D]
        elif 'action' in f:
            gt_actions = f['action'][:]
        else:
            print("Error: Could not find action/joint_action in HDF5")
            return
            
        gt_tensor = torch.tensor(gt_actions)
        
        # 2. Iterate Frames
        # Assuming RoboTwin structure: observation/head_camera/rgb
        if 'observation' in f and 'head_camera' in f['observation'] and 'rgb' in f['observation']['head_camera']:
            rgb_data = f['observation']['head_camera']['rgb'][:]
        else:
            print("Error: Could not find observation/head_camera/rgb in HDF5")
            return
            
        # Optional Mask
        mask_data = None
        if args.use_gt_mask:
            # Check for left_arm_mask
            if 'left_arm_mask' in f['observation']['head_camera']:
                mask_data = f['observation']['head_camera']['left_arm_mask'][:]
            elif 'mask' in f['observation']['head_camera']:
                mask_data = f['observation']['head_camera']['mask'][:]
            else:
                print("Warning: use_gt_mask is True but left_arm_mask/mask not found in HDF5")

        # Processing loop
        episode_pred = []
        T = len(rgb_data)
        
        print(f"Starting inference on {T} frames...")
        
        for i in range(T):
            # Decode RGB
            img_bytes = rgb_data[i]
            if isinstance(img_bytes, (bytes, np.bytes_)):
                 img_np = cv2.imdecode(np.frombuffer(img_bytes.strip(b'\0'), np.uint8), cv2.IMREAD_COLOR)
            else:
                 img_np = img_bytes # Assuming raw if not bytes
                 
            if img_np is None:
                print(f"Error decoding frame {i}")
                continue

            frame_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            input_tensor = preprocessor.process_image(pil_img)
            
            # Mask
            mask_tensor = None
            if mask_data is not None:
                m_bytes = mask_data[i]
                if isinstance(m_bytes, (bytes, np.bytes_)):
                     m_np = cv2.imdecode(np.frombuffer(m_bytes.strip(b'\0'), np.uint8), cv2.IMREAD_UNCHANGED)
                else:
                     m_np = m_bytes
                
                if m_np is not None:
                     if len(m_np.shape) == 2:
                         pil_mask = Image.fromarray(m_np)
                     else:
                         pil_mask = Image.fromarray(cv2.cvtColor(m_np, cv2.COLOR_BGR2RGB))
                         
                     mask_tensor = preprocessor.process_mask(pil_mask)
            
            # Prepare batch
            input_batch = input_tensor.unsqueeze(0).to(accelerator.device)
            if mask_tensor is not None:
                mask_batch = mask_tensor.unsqueeze(0).to(accelerator.device)
            else:
                mask_batch = None
                
            # Inference Logic
            is_mask_model = isinstance(net.model, Mask)
            
            if mask_batch is not None and not is_mask_model:
                masked_images = input_batch * mask_batch
                output = net(masked_images)
            else:
                output = net(input_batch, return_mask=True)
                if isinstance(output, tuple):
                    output, _ = output
            
            if args.use_normalization:
                output = net.normalize(output)
                
            episode_pred.append(output.detach().cpu().numpy())

    if not episode_pred:
        print("Error: No frames processed.")
        return

    # 3. Concatenate and Align
    pred_arr = np.concatenate(episode_pred, axis=0) # [T_video, 14]
    gt_arr = gt_tensor.numpy()                      # [T_action, 14]
    
    # Truncate to the shorter length
    length = min(len(pred_arr), len(gt_arr))
    
    if length == 0:
        print("Error: Length mismatch resulted in 0 overlap.")
        return

    print(f"Frames: {len(pred_arr)}, GT steps: {len(gt_arr)}. Plotting first {length} steps.")
    
    pred_arr = pred_arr[:length]
    gt_arr = gt_arr[:length]
    
    # 4. Plot
    plot_actions(gt_arr, pred_arr, args.save_dir, "single_hdf5_eval")

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
    Process a single video/action pair with optional mask.
    """
    print(f"Processing single video mode.")
    print(f"Video: {args.video_path}")
    print(f"Action: {args.action_path}")
    if args.mask_path:
        print(f"Mask: {args.mask_path}")
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    if not os.path.exists(args.action_path):
        print(f"Error: Action file not found: {args.action_path}")
        return
    if args.use_gt_mask and not args.mask_path:
        print(f"Error: use_gt_mask is True but no mask_path provided")
        return
    if args.mask_path and not os.path.exists(args.mask_path):
        print(f"Error: Mask file not found: {args.mask_path}")
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
    
    # Setup Mask Capture if needed
    cap_mask = None
    if args.mask_path:
        cap_mask = cv2.VideoCapture(args.mask_path)
    
    episode_pred = []
    
    print("Starting inference on video frames...")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocessing: BGR -> RGB -> PIL -> Preprocessor -> Tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        input_tensor = preprocessor.process_image(pil_img)
        # input_tensor: [3, 518, 518]
        
        mask_tensor = None
        if cap_mask is not None:
             ret_mask, frame_mask = cap_mask.read()
             if not ret_mask:
                 print(f"Warning: Mask video ended before main video at frame {frame_idx}")
                 # If mask video is shorter, maybe reusing last mask or dealing with mismatch?
                 # Assuming aligned for now.
             else:
                 frame_mask_rgb = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2RGB)
                 pil_mask = Image.fromarray(frame_mask_rgb)
                 # Expecting preprocessor.process_mask to handle PIL image
                 mask_tensor = preprocessor.process_mask(pil_mask)
                 # mask_tensor: [1, 518, 518]

        # Prepare batch
        input_batch = input_tensor.unsqueeze(0).to(accelerator.device)
        if mask_tensor is not None:
            mask_batch = mask_tensor.unsqueeze(0).to(accelerator.device)
        else:
            mask_batch = None
            
        # Inference Logic (Matching train_idm.py)
        is_mask_model = isinstance(net.model, Mask)
        
        if mask_batch is not None and not is_mask_model:
            # Case 1: We have GT Mask, and Model is NOT Mask type (e.g. ResNet)
            # Apply mask to image
            masked_images = input_batch * mask_batch
            output = net(masked_images)
            # Output is [1, 14]
        else:
            # Case 2: Mask is None OR Model IS Mask type (it predicts mask or ignores GT mask input in this path)
            output = net(input_batch, return_mask=True)
            if isinstance(output, tuple):
                output, _ = output # Ignore predicted mask
            # Output is [1, 14]

        if args.use_normalization:
            output = net.normalize(output)
            
        episode_pred.append(output.detach().cpu().numpy())
        frame_idx += 1

    cap.release()
    if cap_mask is not None:
        cap_mask.release()
    
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
    plot_actions(gt_arr, pred_arr, args.save_dir, "single_mask_eval")


def main():
    args = parse_args()
    seed_torch(1234)
    accelerator = Accelerator()
    
    # Init Model
    net = IDM(model_name=args.model_name, output_dim=14)

    # Load Checkpoint
    try:
        print(f"Loading checkpoint from {args.load_from}")
        loaded_dict = torch.load(args.load_from, map_location='cpu', weights_only=False)
        
        if "model_state_dict" in loaded_dict:
            state_dict = loaded_dict["model_state_dict"]
        else:
            state_dict = loaded_dict
            
        # Load state dict
        # Sometimes keys have 'model.' prefix if saved wrapped, sometimes not.
        # IDM has self.model. 
        # If the saved dict is accelerator.unwrap_model(net).state_dict(), then keys should match IDM structure.
        # Let's try loading.
        missing, unexpected = net.load_state_dict(state_dict, strict=False)
        print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if len(missing) > 0:
            print(f"Missing keys: {missing[:5]} ...")
            
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    net.to(accelerator.device)
    net.eval()

    # Preprocessor
    preprocessor = DinoPreprocessor(args)
    
    # Branch: Single File vs Dataset
    if args.hdf5_path:
        with torch.no_grad():
            process_single_hdf5(args, net, preprocessor, accelerator)
    elif args.video_path and args.action_path:
        # Single File Mode
        with torch.no_grad():
            process_single_video(args, net, preprocessor, accelerator)
            
    elif args.dataset_path:
        pass # Dataset mode implementation omitted/simplified for this new script unless needed
    else:
        print("Error: You must provide (--video_path AND --action_path)")

if __name__ == "__main__":
    main()
