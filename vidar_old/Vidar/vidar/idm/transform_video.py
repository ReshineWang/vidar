import cv2
import numpy as np


def center_crop_to_aspect(frame_bgr: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    先按目标宽高比做中心裁剪（确定性，不随机），返回裁剪后的图（不 resize）。
    - 目标更宽：裁掉上下（减小高度）
    - 目标更窄：裁掉左右（减小宽度）
    """
    h, w = frame_bgr.shape[:2]
    target_ratio = target_w / target_h
    in_ratio = w / h

    if abs(in_ratio - target_ratio) < 1e-8:
        return frame_bgr

    if in_ratio < target_ratio:
        # 输入更窄 -> 想要更宽：只能裁高度
        crop_w = w
        crop_h = int(round(w / target_ratio))
        crop_h = min(crop_h, h)
        top = (h - crop_h) // 2
        left = 0
    else:
        # 输入更宽 -> 想要更窄：裁宽度
        crop_h = h
        crop_w = int(round(h * target_ratio))
        crop_w = min(crop_w, w)
        left = (w - crop_w) // 2
        top = 0

    return frame_bgr[top:top + crop_h, left:left + crop_w]


def temporal_indices_keep_duration(n_src_frames: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """
    把“目标视频时长 + 同时长源视频应覆盖的帧数”的逻辑用于整段视频的 fps 变换：
    - 时长 T = n_src / src_fps
    - 目标总帧数 n_dst = round(T * dst_fps)
    - 第 k 个目标帧对应时间 t=k/dst_fps，取源帧 round(t*src_fps)
    """
    if n_src_frames <= 0:
        return np.array([], dtype=np.int64)
    if src_fps <= 0 or dst_fps <= 0:
        raise ValueError(f"Invalid fps: src_fps={src_fps}, dst_fps={dst_fps}")

    # 目标时长（秒）
    T = n_src_frames / src_fps

    # 目标总帧数（保持时长不变）
    n_dst_frames = int(round(T * dst_fps))
    n_dst_frames = max(n_dst_frames, 1)

    # 均匀采样“时间点”
    t = np.arange(n_dst_frames, dtype=np.float64) / dst_fps  # shape [n_dst]

    # 映射回源帧索引（最近邻时间对齐）
    idx = np.round(t * src_fps).astype(np.int64)

    # 防越界
    idx = np.clip(idx, 0, n_src_frames - 1)
    return idx


def convert_video_30_to_16_with_crop_resize(
    input_path: str,
    output_path: str,
    out_w: int = 832,
    out_h: int = 480,
    dst_fps: float = 16.0,
    interpolation=cv2.INTER_CUBIC,  # BICUBIC
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 0:
        # 有些视频读不到 fps，给个兜底
        src_fps = 30.0

    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_src <= 0:
        # 兜底：逐帧数（慢，但稳）
        frames = []
        while True:
            ret, fr = cap.read()
            if not ret:
                break
            frames.append(fr)
        cap.release()
        n_src = len(frames)
        idx = temporal_indices_keep_duration(n_src, src_fps, dst_fps)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, dst_fps, (out_w, out_h))
        for i in idx:
            frame = frames[int(i)]
            cropped = center_crop_to_aspect(frame, out_w, out_h)
            resized = cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)
            writer.write(resized)
        writer.release()
        print(f"Saved: {output_path}  (src_fps={src_fps:.3f} -> dst_fps={dst_fps:.3f})")
        return

    idx = temporal_indices_keep_duration(n_src, src_fps, dst_fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, dst_fps, (out_w, out_h))

    # 按 index 随机访问某些编码格式可能慢；但最直观、实现最简单
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            # 极少数情况下 set/read 会失败，跳过或用上一帧也行
            continue

        # 1) 先按目标宽高比中心裁剪（确定性）
        cropped = center_crop_to_aspect(frame, out_w, out_h)

        # 2) 再缩放到目标分辨率
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)

        writer.write(resized)

    cap.release()
    writer.release()
    print(f"Saved: {output_path}  (src_fps={src_fps:.3f} -> dst_fps={dst_fps:.3f})")





if __name__ == "__main__":
    convert_video_30_to_16_with_crop_resize(
        input_path="/data/dex/vidar/vidar_old/Vidar/vidar/output/episode0_blocks_ranking_size.mp4",
        output_path="/data/dex/vidar/vidar_old/Vidar/vidar/output/episode0_blocks_ranking_size_resized.mp4",
        out_w=832,
        out_h=480,
        dst_fps=16.0,
    )
