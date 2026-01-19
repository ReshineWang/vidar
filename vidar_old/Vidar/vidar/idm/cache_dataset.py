import os
import re
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


EP_RE = re.compile(r"^episode_?(\d+)\.mp4$")


def parse_episode_id(name: str):
    m = EP_RE.match(name)
    if m is None:
        return None
    return int(m.group(1))


class CacheDataSet(Dataset):
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

        self.video_paths = []
        self.mask_paths = []
        self.qpos_data = []
        self.video_lengths = []

        if self.preprocessor is not None:
            self.preprocessor.set_augmentation_progress(0)

        # ========== 扫描 task ==========
        task_names = [
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]
        print(f"Found {len(task_names)} tasks under {dataset_path}")
        for task in tqdm(task_names, desc="Scanning tasks", disable=disable_pbar):
            demo_root = os.path.join(dataset_path, task, "demo_clean_vidar")
            video_dir = os.path.join(demo_root, "video")
            action_dir = os.path.join(demo_root, "action_gt")

            if not os.path.isdir(video_dir) or not os.path.isdir(action_dir):
                continue

            # 找所有 episode*.mp4（排除 mask）
            video_files = [
                f for f in os.listdir(video_dir)
                if f.endswith(".mp4") and "mask" not in f
            ]

            for vf in video_files:
                ep_id = parse_episode_id(vf)
                if ep_id is None:
                    continue

                # 统一 episode 前缀（兼容 episode0 / episode_0）
                candidates = [
                    f"episode{ep_id}",
                    f"episode_{ep_id}",
                ]

                picked = None
                for ep in candidates:
                    video_path = os.path.join(video_dir, f"{ep}.mp4")
                    qpos_path = os.path.join(action_dir, f"{ep}.pt")
                    mask_path = os.path.join(video_dir, f"{ep}_left_arm_mask.mp4")

                    if not os.path.exists(video_path):
                        continue
                    if not os.path.exists(qpos_path):
                        continue
                    if self.use_gt_mask and not os.path.exists(mask_path):
                        continue

                    picked = (video_path, qpos_path, mask_path)
                    break

                if picked is None:
                    print(f"[Skip] {task}/{vf} : missing qpos or mask")
                    continue

                video_path, qpos_path, mask_path = picked

                # 读取视频长度
                cap = cv2.VideoCapture(video_path)
                video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                if video_len < 30:
                    print(f"[Skip] {video_path} : too short ({video_len})")
                    continue

                qpos = torch.load(qpos_path)
                video_len = min(video_len, len(qpos))

                self.video_paths.append(video_path)
                self.qpos_data.append(qpos)
                self.video_lengths.append(video_len)

                if self.use_gt_mask:
                    self.mask_paths.append(mask_path)

        if len(self.video_lengths) == 0:
            raise RuntimeError(
                f"No valid data found under {dataset_path}\n"
                "Expected structure:\n"
                "  data/{task}/demo_clean_vidar/video/episode*.mp4\n"
                "  data/{task}/demo_clean_vidar/action_gt/episode*.pt"
            )

        self.data_begin = np.cumsum([0] + self.video_lengths[:-1])
        self.data_end = np.cumsum(self.video_lengths)

    def __len__(self):
        return int(self.data_end[-1])

    def _read_frame(self, video_path, idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def __getitem__(self, idx):
        video_idx = int(np.searchsorted(self.data_end, idx, side="right"))
        local_idx = int(idx - self.data_begin[video_idx])

        image = self._read_frame(self.video_paths[video_idx], local_idx)
        mask = None
        if self.use_gt_mask:
            mask = self._read_frame(self.mask_paths[video_idx], local_idx)

        pos = self.qpos_data[video_idx][local_idx]

        if self.preprocessor is not None:
            image = self.preprocessor.process_image(image)
            if mask is not None:
                mask = self.preprocessor.process_mask(mask)
            if np.random.rand() < 0.5:
                image, mask, pos = self.preprocessor.handle_flip(image, mask, pos)

        return image, mask, pos
