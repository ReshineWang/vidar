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
from idm.robotwin_dataset import RoboTwinDataset
from torch.utils.data import Subset
from tqdm import tqdm
import cv2
from PIL import Image
import math

def collate_fn(batch):
    # batch is a list of tuples (image, mask, pos)
    # image is [3, 518, 518], mask is [1, 518, 518] or None, pos is [14]
    images, masks, pos = zip(*batch)
    # preprocess images
    images = torch.stack(images)
    
    if masks[0] is not None:
        masks = torch.stack(masks)
    else:
        masks = None
        
    pos = torch.stack(pos)  # [B, 14]
    return images, masks, pos

def is_close(pos, output):
    limit = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]).to(pos.device)
    # gripper:
    limit[6] = 0.5
    limit[13] = 0.5
    # Handle both single samples and batches
    if pos.dim() == 1:
        return torch.all(torch.abs(pos - output) < limit)
    else:
        return torch.all(torch.abs(pos - output) < limit, dim=1)

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
    parser.add_argument("--load_mp4", action="store_true", default=True)
    parser.add_argument("--use_gt_mask", action="store_true", default=False, help="Use ground truth mask")
    parser.add_argument("--mask_weight", type=float, default=1e-3, help="Mask weight") # Add for compatibility if Preprocessor needs args
    parser.add_argument("--ratio_eval", type=float, default=0.05) # Add for compatibility
    parser.add_argument("--use_transform", action="store_true", default=False, help="Use transform")
    parser.add_argument("--use_normalization", action="store_true", default=False)
    parser.add_argument("--domain", type=str, default="default", help="Dataset domain: default or RoboTwin")
    parser.add_argument("--task_config", type=str, default="demo_clean_vidar", help="Task config subfolder name")
    parser.add_argument("--val_indices", type=str, default=None, help="Path to validation indices file for dataset eval")
    
    # New arguments for single file mode
    parser.add_argument("--video_path", type=str, default="", help="Path to single .mp4 video file")
    parser.add_argument("--action_path", type=str, default="", help="Path to single .pt action file")
    parser.add_argument("--mask_path", type=str, default="", help="Path to single .mp4 mask file (optional unless use_gt_mask is True)")
    parser.add_argument("--hdf5_path", type=str, default="", help="Path to single .hdf5 file")
    parser.add_argument("--num_frames", type=int, default=1, help="Number of frames for input (1, 2, or 3)")
    parser.add_argument("--do_flip", action="store_true", default=False, help="Enable random flip augmentation")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Mixed precision training (no, fp16, bf16)")
    parser.add_argument("--crop_and_resize", action="store_true", default=False, help="Crop and resize input to 832x480")
    parser.add_argument("--compute_smoothness", action="store_true", default=False, help="Compute smoothness reward")

    return parser.parse_args()

import h5py
import io
from idm.transform_video import center_crop_to_aspect


def run_eval_loop(net, dataloader, accelerator, args, loss_fn):
    total_correct = 0
    total_samples = 0
    eval_loss = 0
    all_preds = []
    all_gts = []

    for images, masks, pos in tqdm(dataloader, disable=not accelerator.is_main_process, desc="Evaluating"):
        if args.use_gt_mask:
             masked_images = images * masks
             output = net(masked_images)
        else:
             output = net(images, return_mask=True)
             if isinstance(output, tuple):
                 output, _ = output

        # Calculate batch accuracy using denormalized values
        batch_correct = is_close(pos, output)
        total_correct += batch_correct.sum().item()
        total_samples += len(pos)

        all_preds.append(output.detach().cpu().numpy())
        all_gts.append(pos.detach().cpu().numpy())

        # Loss calculation (optional replication of training metric)
        if args.use_normalization:
             loss = loss_fn(net.normalize(output), net.normalize(pos))
        else:
             loss = loss_fn(output, pos)
             
        eval_loss += loss.item() * len(pos)
        
    avg_loss = eval_loss / total_samples if total_samples > 0 else 0
    acc = total_correct / total_samples if total_samples > 0 else 0
    
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
    else:
        all_preds = np.empty((0, 14))
        all_gts = np.empty((0, 14))
        
    return all_preds, all_gts, avg_loss, acc

def process_dataset(args, net, preprocessor, accelerator):
    # load dataset
    if args.domain == "RoboTwin":
        if accelerator.is_main_process:
            print(f"Using RoboTwinDataset from {args.dataset_path}")
        dataset = RoboTwinDataset(args, dataset_path=args.dataset_path, disable_pbar=not accelerator.is_main_process, preprocessor=preprocessor, use_gt_mask=args.use_gt_mask, type="val", num_frames=args.num_frames, do_flip=args.do_flip)
    else:
        dataset = CacheDataSet(args, dataset_path=args.dataset_path, disable_pbar=not accelerator.is_main_process, preprocessor=preprocessor, use_gt_mask=args.use_gt_mask)
    
    loss_fn = torch.nn.SmoothL1Loss()

    if args.val_indices and os.path.exists(args.val_indices):
        print(f"Loading validation indices from {args.val_indices}")
        try:
             indices = torch.load(args.val_indices)
             dataset = Subset(dataset, indices)
             print(f"Loaded subset with {len(dataset)} samples")
             
             dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False, prefetch_factor=args.prefetch_factor)
             dataloader = accelerator.prepare(dataloader)
             
             print("Starting dataset evaluation (subset)...")
             all_preds, all_gts, avg_loss, acc = run_eval_loop(net, dataloader, accelerator, args, loss_fn)
             print(f"Dataset result: Loss: {avg_loss:.4f}, Correct Rate: {acc:.4f}")
             
             # Plot subset (limit to first 300 frames)
             if len(all_preds) > 0:
                 vis_limit = min(300, len(all_preds))
                 print(f"Plotting visualization for first {vis_limit} samples...")
                 plot_actions(all_gts[:vis_limit], all_preds[:vis_limit], args.save_dir, "dataset_val_subset")

        except Exception as e:
             print(f"Failed to load indices or evaluate subset: {e}")
             return
        
    elif args.val_indices:
        print(f"Warning: val_indices path {args.val_indices} not found.")
        raise FileNotFoundError(f"val_indices file not found: {args.val_indices}")

    elif hasattr(dataset, 'data_begin') and hasattr(dataset, 'data_end'):
        # NEW LOGIC: Plot first 10 individual videos
        print(f"No val_indices provided. Plotting first 10 videos individually from dataset.")
        
        num_videos = min(10, len(dataset.data_end))
        for i in range(num_videos):
            start_idx = int(dataset.data_begin[i])
            end_idx = int(dataset.data_end[i])
            print(f"  - Video {i}: Frames {start_idx} to {end_idx} ({end_idx-start_idx} frames)")
            
            # Create subset for this single video
            video_indices = list(range(start_idx, end_idx))
            subset = Subset(dataset, video_indices)
            
            dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False, prefetch_factor=args.prefetch_factor)
            dataloader = accelerator.prepare(dataloader)
            
            all_preds, all_gts, avg_loss, acc = run_eval_loop(net, dataloader, accelerator, args, loss_fn)
            print(f"    Video {i} Result: Loss={avg_loss:.4f}, Acc={acc:.4f}")
            
            plot_actions(all_gts, all_preds, args.save_dir, f"video_{i:02d}")

            # Smoothness Reward (optional)
            if args.compute_smoothness:
                all_preds_tensor = torch.tensor(all_preds)
                smoothness_reward = compute_smoothness(all_preds_tensor)
                print(f"Computed Smoothness Reward for dataset: {smoothness_reward:.4f}")

    else:
        print(f"No val_indices provided and no video structure found. Evaluating first 400 frames.")
        dataset = Subset(dataset, list(range(min(400, len(dataset)))))

        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn, drop_last=False, prefetch_factor=args.prefetch_factor)
        dataloader = accelerator.prepare(dataloader)
        
        all_preds, all_gts, avg_loss, acc = run_eval_loop(net, dataloader, accelerator, args, loss_fn)
        print(f"Dataset result: Loss: {avg_loss:.4f}, Correct Rate: {acc:.4f}")
        
        if len(all_preds) > 0:
             plot_actions(all_gts, all_preds, args.save_dir, "dataset_first_samples")


        

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
        
        # Pre-load all frames like process_single_video
        video_frames = []
        mask_frames = []
        
        print(f"Pre-loading {T} frames from HDF5...")
        for i in range(T):
            # Decode RGB
            img_bytes = rgb_data[i]
            if isinstance(img_bytes, (bytes, np.bytes_)):
                 img_np = cv2.imdecode(np.frombuffer(img_bytes.strip(b'\0'), np.uint8), cv2.IMREAD_COLOR)
            else:
                 img_np = img_bytes 
                 
            if img_np is None:
                print(f"Error decoding frame {i}, skipping...")
                video_frames.append(None) # Keep index alignment
            else:
                if args.crop_and_resize:
                     target_w, target_h = 832, 480
                     cropped = center_crop_to_aspect(img_np, target_w, target_h)
                     img_np = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                     
                frame_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                video_frames.append(Image.fromarray(frame_rgb))
            
            # Decode Mask
            if mask_data is not None:
                m_bytes = mask_data[i]
                if isinstance(m_bytes, (bytes, np.bytes_)):
                     m_np = cv2.imdecode(np.frombuffer(m_bytes.strip(b'\0'), np.uint8), cv2.IMREAD_UNCHANGED)
                else:
                     m_np = m_bytes
                
                if m_np is not None:
                     if len(m_np.shape) == 2:
                         mask_frames.append(Image.fromarray(m_np))
                     else:
                         mask_frames.append(Image.fromarray(cv2.cvtColor(m_np, cv2.COLOR_BGR2RGB)))
                else:
                     mask_frames.append(None)
        
        print(f"Starting inference on {T} frames with num_frames={args.num_frames}...")
        
        for i in range(T):
            if video_frames[i] is None:
                # If current frame bad, maybe skip or use previous?
                continue
            
            local_idx = i

            # Calculate indices matching RoboTwinDataset logic
            indices_to_load = []
            
            if args.crop_and_resize:

                if args.num_frames == 1:
                    indices_to_load = [local_idx]
                elif args.num_frames == 2:
                    indices_to_load = [local_idx - 2, local_idx]
                elif args.num_frames == 3:
                    indices_to_load = [local_idx - 2, local_idx, local_idx + 2]
                else:
                    raise ValueError(f"Unsupported num_frames: {args.num_frames}")
            else:
                if args.num_frames == 1:
                    indices_to_load = [local_idx]
                elif args.num_frames == 2:
                    indices_to_load = [local_idx - 1, local_idx]
                elif args.num_frames == 3:
                    indices_to_load = [local_idx - 1, local_idx, local_idx + 1]
                else:
                    raise ValueError(f"Unsupported num_frames: {args.num_frames}")

            images_list = []
            for frame_idx in indices_to_load:
                # Clamp index
                clamped_idx = max(0, min(frame_idx, T - 1))
                pil_img = video_frames[clamped_idx]

                input_tensor = preprocessor.process_image(pil_img)
                images_list.append(input_tensor)

            # Stack to create [num_frames, 3, H, W]
            input_volume = torch.stack(images_list, dim=0) 
            
            # Add batch dim -> [1, num_frames, 3, H, W]
            input_batch = input_volume.unsqueeze(0).to(accelerator.device)
            
            # Mask (current frame)
            mask_batch = None
            if mask_data is not None:
                pil_mask = mask_frames[i]
                if pil_mask is not None:
                    mask_tensor = preprocessor.process_mask(pil_mask)
                    mask_batch = mask_tensor.unsqueeze(0).to(accelerator.device)
            else:
                mask_batch = None
                
            # Inference Logic
            is_mask_model = isinstance(net.model, Mask)
            
            if mask_batch is not None and not is_mask_model:
                masked_images = input_batch * mask_batch.unsqueeze(1)
                output = net(masked_images)
            else:
                output = net(input_batch, return_mask=True)
                if isinstance(output, tuple):
                    output, _ = output
            
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

    # 5. Smoothness Reward
    if args.compute_smoothness:
        smoothness_reward = compute_smoothness(torch.tensor(pred_arr))
        print(f"Computed Smoothness Reward: {smoothness_reward:.4f}")


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
        if not np.all(gt_actions[:, i] == 0):
            ax.plot(time_steps, gt_actions[:, i], color='blue', label='Ground Truth', linewidth=1)
        
        # Plot Pred
        if i < pred_actions.shape[1]:
            ax.plot(time_steps, pred_actions[:, i], color='red', label='Predicted', linestyle='--', linewidth=1)
            
        ax.set_title(f'Dimension {i}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')

        # ===== 核心逻辑：条件设置 y 轴范围 =====
        vals = []
        vals.append(gt_actions[:, i])
        if i < pred_actions.shape[1]:
            vals.append(pred_actions[:, i])
        vals = np.concatenate(vals)

        # 如果所有值都在 [-1.2, 1.2] 内 → 固定坐标轴
        if np.all(vals >= -1.2) and np.all(vals <= 1.2):
            ax.set_ylim(-1.2, 1.2)
        # 否则：不设 ylim，用 matplotlib 自动范围

        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    
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
    # if not os.path.exists(args.action_path):
    #     print(f"Error: Action file not found: {args.action_path}")
    #     return
    if args.use_gt_mask and not args.mask_path:
        print(f"Error: use_gt_mask is True but no mask_path provided")
        return
    if args.mask_path and not os.path.exists(args.mask_path):
        print(f"Error: Mask file not found: {args.mask_path}")
        return

    # 1. Load Ground Truth Actions
    gt_tensor = None
    if os.path.exists(args.action_path):
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
    # Read all frames first to support t-1, t+1 logic
    video_frames = []
    cap = cv2.VideoCapture(args.video_path)
    print("Reading video frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if args.crop_and_resize:
             target_w, target_h = 832, 480
             cropped = center_crop_to_aspect(frame, target_w, target_h)
             frame = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
             
        # Preprocessing: BGR -> RGB -> PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frames.append(Image.fromarray(frame_rgb))
    cap.release()

    if not video_frames:
        print("Error: No frames read from video.")
        return

    video_masks = []
    if args.mask_path:
        cap_mask = cv2.VideoCapture(args.mask_path)
        print("Reading mask frames...")
        while cap_mask.isOpened():
            ret, frame = cap_mask.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_masks.append(Image.fromarray(frame_rgb))
        cap_mask.release()

    episode_pred = []
    T = len(video_frames)
    print(f"Starting inference on {T} frames with num_frames={args.num_frames}...")
    
    for i in range(T):
        local_idx = i

        # Calculate indices matching RoboTwinDataset logic
        indices_to_load = []
        
        if args.crop_and_resize:

            if args.num_frames == 1:
                indices_to_load = [local_idx]
            elif args.num_frames == 2:
                indices_to_load = [local_idx - 2, local_idx]
            elif args.num_frames == 3:
                indices_to_load = [local_idx - 2, local_idx, local_idx + 2]
            else:
                raise ValueError(f"Unsupported num_frames: {args.num_frames}")
        else:
            if args.num_frames == 1:
                indices_to_load = [local_idx]
            elif args.num_frames == 2:
                indices_to_load = [local_idx - 1, local_idx]
            elif args.num_frames == 3:
                indices_to_load = [local_idx - 1, local_idx, local_idx + 1]
            else:
                raise ValueError(f"Unsupported num_frames: {args.num_frames}")
            

        images_list = []
        for frame_idx in indices_to_load:
            # Clamp index
            clamped_idx = max(0, min(frame_idx, T - 1))
            pil_img = video_frames[clamped_idx]
            
            input_tensor = preprocessor.process_image(pil_img)
            images_list.append(input_tensor)

        # Stack to create [num_frames, 3, H, W]
        input_volume = torch.stack(images_list, dim=0) 
        
        # Add batch dim -> [1, num_frames, 3, H, W]
        input_batch = input_volume.unsqueeze(0).to(accelerator.device)

        # Mask logic (current frame only)
        mask_batch = None
        if i < len(video_masks):
            pil_mask = video_masks[i]
            mask_tensor = preprocessor.process_mask(pil_mask)
            mask_batch = mask_tensor.unsqueeze(0).to(accelerator.device)
            # mask_batch: [1, 1, H, W] (typically dataset returns [1, H, W], unsqueeze gives [1, 1, H, W])
            # Wait, preprocessor.process_mask returns [1, H, W]
            # So unsqueeze(0) gives [1, 1, H, W]. 
            # If Model expects [1, 1, H, W] for mask? 
            # In collate_fn: masks = torch.stack(masks) -> [B, 1, H, W]
            pass
        elif args.mask_path:
             # Mask video shorter than video
             pass

        # Inference Logic
        is_mask_model = isinstance(net.model, Mask)
        
        if mask_batch is not None and not is_mask_model:
            # Case 1: We have GT Mask, and Model is NOT Mask type (e.g. ResNet)
            # Apply mask to image
            
            # Mask is usually for the query frame (last frame or middle frame?)
            # In IDM/RoboTwinDataset, mask is for the 'current' frame (local_idx).
            # The input images are [1, T, 3, H, W]. Mask is [1, 1, H, W].
            # We need to broadcast mask or apply it to all frames?
            # RoboTwinDataset: "masked_images = images * masks" in train_idm.py (line 351).
            # If images is [B, T, 3, H, W] and masks is [B, 1, 518, 518] (from collate).
            # Broadcasting should work: [B, T, 3, H, W] * [B, 1, 1, H, W] (unsqueeze dims).
            # collate_fn: masks = torch.stack(masks). masks[0] is [1, 518, 518] (channels, h, w) or [518, 518]?
            # process_mask: returns [1, H, W].
            # collate: [B, 1, H, W].
            # training: masked_images = images * masks. 
            # images [B, T, 3, H, W]. masks [B, 1, H, W].
            # We need masks to be [B, 1, 1, H, W] to broadcast over T and C?
            # Or [B, 1, 1, H, W] to broadcast over T=1?
            
            # Let's trust broadcasting logic or unsqueeze appropriately.
            # Here: mask_batch is [1, 1, H, W].
            # input_batch is [1, T, 3, H, W].
            # mask_batch.unsqueeze(1) -> [1, 1, 1, H, W]. Broadcasts to [1, T, 3, H, W].
            
            masked_images = input_batch * mask_batch.unsqueeze(1)
            output = net(masked_images)
            # Output is [1, 14]
        else:
            # Case 2: Mask is None OR Model IS Mask type
            output = net(input_batch, return_mask=True)
            if isinstance(output, tuple):
                output, _ = output # Ignore predicted mask
            # Output is [1, 14]

        episode_pred.append(output.detach().cpu().numpy())


    if gt_tensor is not None:
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
    
    else:
        pred_arr = np.concatenate(episode_pred, axis=0) # [T_video, 14]
        print(f"No action file provided. Predicted {len(pred_arr)} steps. Plotting predictions only.")
        plot_actions(np.zeros_like(pred_arr), pred_arr, args.save_dir, "single_mask_eval_no_gt")

    # Optional: Compute smoothness reward
    if args.compute_smoothness:
        action_tensor = torch.tensor(pred_arr)
        smoothness_reward = compute_smoothness(action_tensor)
        print(f"Computed Smoothness Reward: {smoothness_reward:.4f}")


def compute_smoothness(actions):
    """
    Smoothness reward for 14D actions.

    actions: [T, 14]
    - joint dims are radians
    - dim 6 and 13 are grippers in [0,1]
    - fps = 16Hz => dt = 1/16

    Returns:
      score (float): higher is smoother, in [0, 10] approximately.
    """
    if actions.shape[0] < 4:
        return 0.0

    dt = 1.0 / 16.0
    T, D = actions.shape
    device, dtype = actions.device, actions.dtype

    # ---- dims ----
    grip_dims = [6, 13]

    # ---- per-dim speed limits (rad/s) ----
    # PiPER joint max speed (deg/s): J1 180, J2 195, J3 180, J4-6 225
    # -> rad/s: 3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270
    v_arm = torch.tensor([3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270],
                         device=device, dtype=dtype)
    safety = 0.85
    v_arm = v_arm * safety

    v_max = torch.zeros(D, device=device, dtype=dtype)
    v_max[0:6] = v_arm
    v_max[7:13] = v_arm
    v_max[6] = 2.0   # gripper units/s
    v_max[13] = 2.0

    # ---- per-dim accel limits (rad/s^2) ----
    a_max = torch.full((D,), 5.0, device=device, dtype=dtype)
    a_max[6] = 10.0  # gripper units/s^2
    a_max[13] = 10.0

    # ---- finite differences ----
    v = (actions[1:] - actions[:-1]) / dt      # [T-1, D]
    a = (v[1:] - v[:-1]) / dt                  # [T-2, D]
    j = (a[1:] - a[:-1]) / dt                  # [T-3, D]

    # ---- huber ----
    def huber(x, delta):
        ax = x.abs()
        return torch.where(ax <= delta, 0.5 * (x ** 2), delta * (ax - 0.5 * delta))

    # ---- dim weights ----
    w = torch.ones(D, device=device, dtype=dtype)
    w[6] = 0.2
    w[13] = 0.2
    w = w / (w.mean() + 1e-8)

    # ---- huber thresholds ----
    delta_a = a_max * 0.5                 # [D]
    delta_j = (a_max / dt) * 0.5          # [D]  (since jerk ~ Δa/dt)

    # ---- energy penalties ----
    # broadcast: a/j shape [..., D], delta_*/w shape [D]
    acc_pen  = (huber(a, delta_a) * w).mean()
    jerk_pen = (huber(j, delta_j) * w).mean()

    # ---- soft-limit penalties ----
    v_violate = torch.relu(v.abs() - v_max)
    a_violate = torch.relu(a.abs() - a_max)
    vlim_pen = ((v_violate ** 2) * w).mean()
    alim_pen = ((a_violate ** 2) * w).mean()

    # ---- total penalty ----
    penalty_raw = (
        0.5 * jerk_pen +
        0.25 * acc_pen +
        2.0 * vlim_pen +
        1.0 * alim_pen
    )

    # ---- debug prints (opt-in) ----
    # Set env: DEBUG_SMOOTHNESS=1 to print
    with torch.no_grad():
        # basic stats to see scale
        v_max_abs = v.abs().max().item()
        a_max_abs = a.abs().max().item()
        j_max_abs = j.abs().max().item()

        v_mean = v.abs().mean().item()
        a_mean = a.abs().mean().item()
        j_mean = j.abs().mean().item()

        print(
            "[DEBUG] Smoothness components:\n"
            f"  penalty_raw = {penalty_raw.item():.4f}\n"
            f"    jerk_pen  = {jerk_pen.item():.4f}\n"
            f"    acc_pen   = {acc_pen.item():.4f}\n"
            f"    vlim_pen  = {vlim_pen.item():.4f}\n"
            f"    alim_pen  = {alim_pen.item():.4f}\n"
            f"  stats:\n"
            f"    max|v|={v_max_abs:.4f}, mean|v|={v_mean:.4f}\n"
            f"    max|a|={a_max_abs:.4f}, mean|a|={a_mean:.4f}\n"
            f"    max|j|={j_max_abs:.4f}, mean|j|={j_mean:.4f}\n"
        )

    P = penalty_raw

    # reference scale: roughly "good smoothness" level
    P0 = 2000.0      # try 2000 ~ 5000

    gamma = 0.5      # 0.5 is very safe for GRPO
    MaxReward = 10.0

    score = MaxReward * torch.pow(1.0 + P / P0, -gamma)

    return score.item()


        

def main():
    args = parse_args()
    seed_torch(1234)
    accelerator = Accelerator()
    
    # Init Model
    net = IDM(model_name=args.model_name, output_dim=14, num_frames=args.num_frames)

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
    elif args.video_path or args.action_path:
        # Single File Mode
        with torch.no_grad():
            process_single_video(args, net, preprocessor, accelerator)
            
    elif args.dataset_path:
        with torch.no_grad():
            process_dataset(args, net, preprocessor, accelerator)
    else:
        print("Error: You must provide --video_path OR --action_path OR --hdf5_path OR --dataset_path to evaluate.")

if __name__ == "__main__":
    main()
