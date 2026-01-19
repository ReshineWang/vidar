import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm

class RoboTwinDataset(Dataset):
    def __init__(
        self,
        args,
        dataset_path,
        disable_pbar=False,
        type="train",
        preprocessor=None,
        use_gt_mask=False,
    ):
        self.dataset_path = dataset_path
        self.type = type
        self.use_gt_mask = use_gt_mask
        self.preprocessor = preprocessor

        # Default keys for RoboTwin
        self.action_key = "joint_action/vector"
        self.rgb_key = "observation/head_camera/rgb"
        self.mask_key = "observation/head_camera/left_arm_mask"
        self.task_config = getattr(args, 'task_config', 'demo_clean_vidar')

        self.h5_paths = []
        self.lengths = []
        self.actions = []

        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        # Only include .hdf5 files that live under each task's `self.task_config` subfolder.
        # e.g. /dataset_path/<task_name>/<self.task_config>/.../*.hdf5
        h5_files = []
        for task_name in os.listdir(dataset_path):
            task_dir = os.path.join(dataset_path, task_name)
            if not os.path.isdir(task_dir):
                continue
            cfg_dir = os.path.join(task_dir, self.task_config)
            if not os.path.exists(cfg_dir):
                # skip tasks that don't have the configured subfolder
                continue
            for root, _, files in os.walk(cfg_dir):
                for f in files:
                    if f.endswith('.hdf5'):
                        h5_files.append(os.path.join(root, f))
        
        h5_files.sort() # Ensure deterministic order
        
        if not disable_pbar:
            print(f"Found {len(h5_files)} HDF5 files in {dataset_path} for domain RoboTwin")
        
        valid_files = 0
        for h5_path in tqdm(h5_files, desc="Loading RoboTwin metadata", disable=disable_pbar):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if self.action_key not in f:
                        continue
                    
                    if self.rgb_key not in f:
                         continue
                        
                    rgb_ds = f[self.rgb_key]
                    action_ds = f[self.action_key]
                    
                    # Assuming actions and images are aligned. Use min length just in case.
                    length = min(len(rgb_ds), len(action_ds))
                    
                    if length == 0:
                        continue
                        
                    # Preload actions (Assuming they fit in memory, e.g. 50 episodes * 1000 steps * 14 floats is small)
                    action_data = action_ds[:length] 
                    
                    if self.use_gt_mask:
                        if self.mask_key not in f:
                             # If GT mask is required but missing, skip this file
                             continue

                    self.h5_paths.append(h5_path)
                    self.lengths.append(length)
                    self.actions.append(torch.from_numpy(action_data).float())
                    valid_files += 1
                    
            except Exception as e:
                print(f"Error reading {h5_path}: {e}")
                continue

        if valid_files == 0:
            if not disable_pbar:
                 print(f"Warning: No valid HDF5 data found in {dataset_path} with keys {self.action_key}, {self.rgb_key}")
            if len(h5_files) == 0:
                 raise RuntimeError(f"No .hdf5 files found in {dataset_path}")
            else:
                 raise RuntimeError(f"Found {len(h5_files)} .hdf5 files but none had valid keys ({self.action_key}, {self.rgb_key})")

        self.data_begin = np.cumsum([0] + self.lengths[:-1])
        self.data_end = np.cumsum(self.lengths)
        if not disable_pbar:
             print(f"Total valid videos: {valid_files}, Total frames: {self.data_end[-1]}")
        
    def __len__(self):
        return int(self.data_end[-1])

    def _decode_image(self, data):
        # Handle encoded image bytes -> RGB PIL Image
        # Check if it is bytes
        if isinstance(data, (bytes, bytearray, np.bytes_)):
            if isinstance(data, np.bytes_):
                data = bytes(data)
            # imdecode returns BGR
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image bytes")
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        elif isinstance(data, np.ndarray):
            # Already decoded or raw array
            return Image.fromarray(data)
        return None

    def _decode_mask(self, data):
        # Handle mask -> PIL Image
        # Assuming data is raw numpy array (H, W) or (H, W, 1) or (H, W, 3)
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                return Image.fromarray(data)
            elif data.ndim == 3:
                if data.shape[2] == 1:
                    return Image.fromarray(data[:, :, 0])
                else:
                    return Image.fromarray(data)
        
        # If encoded bytes
        if isinstance(data, (bytes, bytearray, np.bytes_)):
             if isinstance(data, np.bytes_):
                data = bytes(data)
             img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
             return Image.fromarray(img)

        return None

    def __getitem__(self, idx):
        video_idx = int(np.searchsorted(self.data_end, idx, side="right"))
        local_idx = int(idx - self.data_begin[video_idx])
        
        h5_path = self.h5_paths[video_idx]
        
        # Open file on demand
        try:
            with h5py.File(h5_path, 'r') as f:
                img_data = f[self.rgb_key][local_idx]
                image = self._decode_image(img_data)
                
                mask = None
                if self.use_gt_mask:
                    mask_data = f[self.mask_key][local_idx]
                    mask = self._decode_mask(mask_data)
        except Exception as e:
            # print(f"Error reading frame {idx} (local {local_idx}) from {h5_path}: {e}")
            raise RuntimeError(f"Failed to read frame at {h5_path} index {local_idx}: {e}")

        # Get action from memory
        pos = self.actions[video_idx][local_idx]

        if self.preprocessor is not None:
            image = self.preprocessor.process_image(image)
            if mask is not None:
                mask = self.preprocessor.process_mask(mask)
            # Apply flip augmentation if needed
            if np.random.rand() < 0.5:
                image, mask, pos = self.preprocessor.handle_flip(image, mask, pos)

        return image, mask, pos
