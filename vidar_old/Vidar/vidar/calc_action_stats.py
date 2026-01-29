import os
import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm


def iter_h5_files(dataset_path: str, task_config: str):
    """
    Only include .hdf5 files under:
      dataset_path/<task_name>/<task_config>/**.hdf5
    """
    h5_files = []
    for task_name in os.listdir(dataset_path):
        task_dir = os.path.join(dataset_path, task_name)
        if not os.path.isdir(task_dir):
            continue

        cfg_dir = os.path.join(task_dir, task_config)
        if not os.path.exists(cfg_dir):
            continue

        for root, _, files in os.walk(cfg_dir):
            for f in files:
                if f.endswith(".hdf5"):
                    h5_files.append(os.path.join(root, f))

    h5_files.sort()
    return h5_files


class RunningStats:
    """
    Welford online mean/std for vectors.
    Tracks population variance (ddof=0); for training normalization it's fine.
    """
    def __init__(self):
        self.n = 0
        self.mean = None
        self.M2 = None  # sum of squared deviations

    def update(self, x: np.ndarray):
        """
        x: shape [T, D] float32/float64 numpy array
        """
        if x.ndim == 1:
            x = x[None, :]
        x = x.astype(np.float64, copy=False)

        if self.mean is None:
            d = x.shape[1]
            self.mean = np.zeros((d,), dtype=np.float64)
            self.M2 = np.zeros((d,), dtype=np.float64)

        for i in range(x.shape[0]):
            self.n += 1
            delta = x[i] - self.mean
            self.mean += delta / self.n
            delta2 = x[i] - self.mean
            self.M2 += delta * delta2

    def finalize(self):
        if self.n == 0:
            raise RuntimeError("No samples were accumulated.")
        var = self.M2 / self.n  # population variance
        std = np.sqrt(var)
        return self.mean.astype(np.float32), std.astype(np.float32), self.n


def calculate_stats_hdf5(
    dataset_path: str,
    task_config: str,
    action_key: str,
    rgb_key: str | None = None,
    align_with_rgb: bool = True,
):
    print(f"Scanning RoboTwin dataset at: {dataset_path}")
    print(f"task_config: {task_config}")
    print(f"action_key : {action_key}")
    if rgb_key is not None:
        print(f"rgb_key    : {rgb_key}")
        print(f"align_with_rgb: {align_with_rgb}")

    h5_files = iter_h5_files(dataset_path, task_config)
    if not h5_files:
        raise RuntimeError(f"No .hdf5 files found under {dataset_path}/*/{task_config}/**/*.hdf5")

    print(f"Found {len(h5_files)} HDF5 files. Calculating statistics...")

    stats = RunningStats()
    valid = 0
    skipped = 0

    for h5_path in tqdm(h5_files, desc="Reading actions"):
        try:
            with h5py.File(h5_path, "r") as f:
                if action_key not in f:
                    skipped += 1
                    continue

                action_ds = f[action_key]

                if align_with_rgb and rgb_key is not None:
                    if rgb_key not in f:
                        skipped += 1
                        continue
                    rgb_ds = f[rgb_key]
                    length = min(len(action_ds), len(rgb_ds))
                else:
                    length = len(action_ds)

                if length <= 0:
                    skipped += 1
                    continue

                # Read actions [T, D]
                action = action_ds[:length]
                if not isinstance(action, np.ndarray):
                    action = np.array(action)

                # Ensure float
                action = action.astype(np.float32, copy=False)

                # Update running stats
                stats.update(action)
                valid += 1

        except Exception as e:
            print(f"[WARN] Error reading {h5_path}: {e}")
            skipped += 1

    if valid == 0:
        raise RuntimeError(
            f"Found {len(h5_files)} .hdf5 files, but none had valid key '{action_key}' "
            f"(and rgb alignment={align_with_rgb})."
        )

    mean, std, n = stats.finalize()

    # avoid division-by-zero for constant dims
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    print(f"\nValid files: {valid}, Skipped files: {skipped}")
    print(f"Total samples: {n}")
    print(f"Dimension: {mean.shape[0]}")

    print("\n" + "=" * 50)
    print("COPY AND PASTE THE FOLLOWING INTO idm.py:")
    print("=" * 50)

    mean_str = ", ".join([f"{x:.8f}" for x in mean.tolist()])
    std_str = ", ".join([f"{x:.8f}" for x in std.tolist()])

    print(f"train_mean = torch.tensor([{mean_str}])")
    print(f"train_std = torch.tensor([{std_str}])")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path of RoboTwin dataset")
    parser.add_argument("--task_config", type=str, default="demo_clean_vidar", help="Subfolder under each task")
    parser.add_argument("--action_key", type=str, default="joint_action/vector", help="HDF5 key for actions")
    parser.add_argument(
        "--rgb_key",
        type=str,
        default="observation/head_camera/rgb",
        help="Optional: align action length with rgb length (min of both).",
    )
    parser.add_argument(
        "--no_rgb_align",
        action="store_true",
        help="If set, do NOT align with rgb length; use full action length.",
    )
    args = parser.parse_args()

    calculate_stats_hdf5(
        dataset_path=args.dataset_path,
        task_config=args.task_config,
        action_key=args.action_key,
        rgb_key=args.rgb_key,
        align_with_rgb=(not args.no_rgb_align),
    )
