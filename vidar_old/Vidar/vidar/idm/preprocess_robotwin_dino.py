import os
import sys
import argparse
import h5py
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# Add project root to sys.path to allow imports from idm
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from idm.idm import DinoResNet
from idm.preprocessor import DinoPreprocessor
from idm.robotwin_dataset import RoboTwinDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess RoboTwin dataset images to DinoV3 features")
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of RoboTwin dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Root path to save dino features")
    parser.add_argument("--task_config", type=str, default="demo_clean_vidar", help="Task config subfolder name to look for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    # Arguments required by DinoPreprocessor/DinoResNet initialization but not strictly used for logic here
    parser.add_argument("--use_transform", action="store_true", default=False, help="Use data augmentation (Shared with preprocessor)")
    
    return parser.parse_args()

def check_keys(h5_path, keys):
    try:
        with h5py.File(h5_path, 'r') as f:
            for k in keys:
                if k not in f:
                    return False
            return True
    except:
        return False

def decode_image_batch(img_data_batch):
    """
    Decodes a batch of image data (bytes or numpy arrays) to PIL Images.
    """
    images = []
    for data in img_data_batch:
        if isinstance(data, (bytes, bytearray, np.bytes_)):
            if isinstance(data, np.bytes_):
                data = bytes(data)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(img))
            else:
                 # Should not happen in clean dataset, but handle gracefuuly?
                 # Or just append a black image to keep batch alignment?
                 # Raising error is safer.
                 raise ValueError("Failed to decode image")
        elif isinstance(data, np.ndarray):
             images.append(Image.fromarray(data))
    return images

def main():
    args = parse_args()
    
    print(f"Dataset Path: {args.dataset_path}")
    print(f"Output Path: {args.output_path}")
    print(f"Device: {args.device}")

    # 1. Discover Files (Reusing logic style from RoboTwinDataset)
    h5_files = []
    task_dirs = sorted(os.listdir(args.dataset_path))
    
    print("Scanning for HDF5 files...")
    for task_name in task_dirs:
        task_dir = os.path.join(args.dataset_path, task_name)
        if not os.path.isdir(task_dir):
            continue
            
        cfg_dir = os.path.join(task_dir, args.task_config)
        if not os.path.exists(cfg_dir):
            continue
            
        for root, _, files in os.walk(cfg_dir):
            for f in files:
                if f.endswith('.hdf5'):
                    h5_path = os.path.join(root, f)
                    h5_files.append(h5_path)
    
    print(f"Found {len(h5_files)} files.")
    if len(h5_files) == 0:
        return

    # 2. Load Model
    print("Loading DINOv3 Backbone...")
    # Initialize DinoResNet to get the backbone. We only need the backbone.
    # Note: DinoResNet loads weights internally.
    dino_model = DinoResNet(output_dim=14) 
    backbone = dino_model.backbone
    backbone.to(args.device)
    backbone.eval()
    
    # 3. Initialize Preprocessor
    # Force use_transform=False to avoid random augmentation during preprocessing
    args.use_transform = False 
    preprocessor = DinoPreprocessor(args)

    # 4. Process Files
    rgb_key = "observation/head_camera/rgb" # Hardcoded based on RoboTwinDataset

    for h5_path in tqdm(h5_files, desc="Processing Videos"):
        # Calculate relative path to maintain structure
        rel_path = os.path.relpath(h5_path, args.dataset_path)
        out_file_path = os.path.join(args.output_path, rel_path)
        
        # Skip if already exists? Or user might want to overwrite. 
        # Making it safe: if exists and valid, maybe skip. But easier to just overwrite.
        if os.path.exists(out_file_path):
             continue

        os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
        
        try:
            with h5py.File(h5_path, 'r') as fin:
                if rgb_key not in fin:
                    print(f"Skipping {h5_path}: Missing key {rgb_key}")
                    continue
                
                rgb_ds = fin[rgb_key]
                total_frames = len(rgb_ds)
                
                all_features = []
                
                # Batch Processing
                for i in range(0, total_frames, args.batch_size):
                    batch_end = min(i + args.batch_size, total_frames)
                    
                    # Read batch from HDF5
                    # Note: HDF5 slicing returns numpy array
                    raw_batch = rgb_ds[i:batch_end]
                    
                    # Decode images
                    pil_images = decode_image_batch(raw_batch)
                    
                    # Preprocess
                    # process_image returns Tensor [3, 512, 512]
                    # We stack them -> [B, 3, 512, 512]
                    tensor_batch = torch.stack([preprocessor.process_image(img) for img in pil_images])
                    tensor_batch = tensor_batch.to(args.device)
                    
                    # Forward Pass
                    with torch.no_grad():
                        features_dict = backbone.forward_features(tensor_batch)
                        # 'x_norm_patchtokens': [B, N, C]
                        patch_tokens = features_dict['x_norm_patchtokens']
                        
                        B, N, C = patch_tokens.shape
                        H_grid = W_grid = int(N ** 0.5)
                        
                        # Reshape to [B, C, H, W] to be ready for ResNet consumption
                        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
                        all_features.append(feature_map.cpu().numpy())
                
                if all_features:
                    final_features = np.concatenate(all_features, axis=0) # [Total_Frames, C, H, W]
                    
                    # Save to new HDF5
                    # Using atomic write (write to tmp then move) is better practice but direct write is simple here
                    with h5py.File(out_file_path, 'w') as fout:
                        fout.create_dataset("features", data=final_features, compression="gzip")
        
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
            # Clean up partial file
            if os.path.exists(out_file_path):
                os.remove(out_file_path)

if __name__ == "__main__":
    main()
