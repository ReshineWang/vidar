import os
import h5py
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from idm.transform_video import center_crop_to_aspect

class RoboTwinDataset(Dataset):
    def __init__(
        self,
        args,
        dataset_path,
        disable_pbar=False,
        type="train",
        preprocessor=None,
        use_gt_mask=False,
        num_frames=1,
        do_flip=False,
    ):
        self.dataset_path = dataset_path
        self.type = type
        self.use_gt_mask = use_gt_mask
        self.preprocessor = preprocessor
        self.num_frames = num_frames
        self.do_flip = do_flip
        self.crop_and_resize = getattr(args, 'crop_and_resize', False)

        # Default keys for RoboTwin
        self.action_key = "joint_action/vector"
        self.rgb_key = "observation/head_camera/rgb"
        self.mask_key = "observation/head_camera/left_arm_mask"
        self.task_config = getattr(args, 'task_config', 'demo_clean')

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
            
            if self.crop_and_resize:
                target_w, target_h = 832, 480
                cropped = center_crop_to_aspect(img, target_w, target_h)
                img = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return Image.fromarray(img)
        elif isinstance(data, np.ndarray):
            # Already decoded or raw array
            # Assuming BGR for consistency if coming from cv2, but h5 usually stores RGB or encoded bytes.
            # If it's a raw array it's ambiguous. But _decode_image usually handles bytes from HDF5.
            # If we assume it's like the bytes path:
            if self.crop_and_resize:
                 target_w, target_h = 832, 480
                 # Caution: center_crop_to_aspect assumes HWC.
                 cropped = center_crop_to_aspect(data, target_w, target_h)
                 data = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                 
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
                rgb_ds = f[self.rgb_key]
                if self.use_gt_mask:
                    mask_ds = f[self.mask_key]

                # Determine indices to load based on num_frames
                indices_to_load = []
                if self.num_frames == 1:
                    indices_to_load = [local_idx]
                elif self.num_frames == 2:
                    # t-2, t
                    indices_to_load = [local_idx - 2, local_idx]
                elif self.num_frames == 3:
                     # t-2, t, t+2
                    indices_to_load = [local_idx - 2, local_idx, local_idx + 2]
                else:
                    raise ValueError(f"Unsupported num_frames: {self.num_frames}")

                # Handle padding by clamping
                video_len = len(rgb_ds) # Or use self.lengths[video_idx]
                images_list = []
                
                # We only need mask for the CURRENT frame (local_idx) usually
                # But let's check if we want mask sequence. Usually mask is for single frame supervision.
                mask = None
                if self.use_gt_mask:
                    mask_data = mask_ds[local_idx]
                    mask = self._decode_mask(mask_data)
                
                # Random flip decision (consistent across frames)
                is_flipped = False
                if self.do_flip and self.preprocessor is not None and self.type == 'train' and np.random.rand() < 0.5:
                    is_flipped = True

                for i, frame_idx in enumerate(indices_to_load):
                    # Clamp index
                    clamped_idx = max(0, min(frame_idx, video_len - 1))
                    
                    img_data = rgb_ds[clamped_idx]
                    image = self._decode_image(img_data)
                    
                    if self.preprocessor is not None:
                        image = self.preprocessor.process_image(image)
                        # Flip here if decided
                        if is_flipped:
                             # image is [3, H, W]
                             image = torch.flip(image, [-1])
                    
                    images_list.append(image)

                if mask is not None and self.preprocessor is not None:
                    mask = self.preprocessor.process_mask(mask)
                    if is_flipped:
                        mask = torch.flip(mask, [-1])

        except Exception as e:
            # print(f"Error reading frame {idx} (local {local_idx}) from {h5_path}: {e}")
            raise RuntimeError(f"Failed to read frame at {h5_path} index {local_idx}: {e}")

        # Get action from memory (for current frame)
        pos = self.actions[video_idx][local_idx]
        if is_flipped:
             # Manually flip pos to avoid dummy tensor shape issues
             # Logic copied from preprocessor.handle_flip
             flipped_pos = torch.zeros_like(pos)
             flipped_pos[:7] = pos[7:]
             flipped_pos[7:] = pos[:7]
             # Negate specific components that need to be reversed (based on ALOHA/RoboTwin convention)
             flipped_pos[[0, 4, 5, 7, 11, 12]] *= -1
             pos = flipped_pos

        # Stack or return single
        # if self.num_frames > 1:
        #     final_image = torch.stack(images_list) # [T, 3, H, W]
        # else:
        #     final_image = images_list[0] # [3, H, W]
        final_image = torch.stack(images_list, dim=0) # [T, 3, H, W]


        return final_image, mask, pos
