import os
import sys

# Change importing path to use Vidar IDM logic
sys.path.append("/data/dex/vidar/vidar_old/Vidar/vidar")

import torch
import hashlib
from base64 import b64encode
from fastapi import FastAPI
from pydantic import BaseModel
import json
import torchvision
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS

# Use original IDM imports
from idm.idm import IDM
from idm.transform_video import center_crop_to_aspect
from idm.preprocessor import DinoPreprocessor

import io
import base64
from PIL import Image
import logging
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Request(BaseModel):
    prompt: str
    imgs: list
    num_conditional_frames: int = 1
    num_new_frames: int = 16
    seed: int = 1234
    num_sampling_step: int = 5
    guide_scale: float = 5.0
    password: str = ""
    return_imgs: bool = False
    clean_cache: bool = False
    gt_action_path: str = ""
    mode: str = "vidar" # vidar, gt_action, idm_action
    use_transform: bool = False # NEW: align with eval_idm.py

class PreprocessArgs:
    """Mock args for DinoPreprocessor"""
    use_transform = False

def sha256(text):
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()

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
    # T, D = actions.shape
    device = actions.device
    dtype = actions.dtype
    
    # Ensure tensor
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions, device=device, dtype=dtype)
        
    T, D = actions.shape

    # ---- dims ----
    # grip_dims = [6, 13]

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
        abs_x = x.abs()
        mask = abs_x <= delta
        return torch.where(mask, 0.5 * x.pow(2), delta * (abs_x - 0.5 * delta))

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

    P = penalty_raw

    # reference scale: roughly "good smoothness" level
    P0 = 2000.0      # try 2000 ~ 5000

    gamma = 0.5      # 0.5 is very safe for GRPO
    MaxReward = 10.0

    score = MaxReward * torch.pow(1.0 + P / P0, -gamma)

    return score.item()


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


def init():
    global wan_ti2v
    global ulysses_size
    global cfg
    global processor
    global mask_processor
    global idm
    global dino_preprocessor # NEW
    
    # 硬编码配置或从环境变量读取
    cfg = WAN_CONFIGS["ti2v-5B"]
    
    logger.info(f"Current PID: {os.getpid()}")
    # 单卡模式下，Rank相关默认为0/1
    rank = int(os.getenv("RANK", 0))
    pt_dir = os.getenv("MODEL", None)
    idm_path = os.getenv("IDM", None)

    # 默认路径
    default_idm_path = "/data/dex/vidar/vidar_ckpts/resnet_plus_robotwin/big_view.pt"
    if not idm_path:
        idm_path = default_idm_path
        
    model_name = "resnet_plus"
    if "resnet_plus" in idm_path or "big_view" in idm_path:
        model_name = "resnet_plus"
    
    # Allow override
    model_name = os.getenv("IDM_MODEL_NAME", model_name)
    
    # 直接使用 CUDA_VISIBLE_DEVICES 里的第一个设备 (即 cuda:0)
    device = 0
    
    # 初始化 idm/DinoPreprocessor
    # Use DinoPreprocessor exactly as eval_idm.py does
    args = PreprocessArgs()
    # If users enable use_transform logic, they can set it later, 
    # but DinoPreprocessor initializes normalization buffers. 
    # Transforms can be updated dynamically if we passed dynamic args, but let's stick to default.
    dino_preprocessor = DinoPreprocessor(args)
    logger.info("Initialized DinoPreprocessor")
    
    # Use preprocessor as processor interface if needed, or keep processor for fallback
    # For compatibility, we can keep processor but prefer dino_preprocessor for logic
    
    # Manual fallback setup if needed (not strictly used if we rely on dino_preprocessor)
    if model_name == "resnet_plus":
        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((518, 518)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    mask_processor = torchvision.transforms.Resize((736, 640))
    
    # 加载 IDM 模型
    try:
        output_dim = 14
        if "_out.pt" in idm_path:
            output_dim = int(idm_path.split("_out.pt")[0].split("_")[-1])
        idm = IDM(model_name=model_name, output_dim=output_dim, num_frames=3).to(device)
    except Exception as e:
        logger.error(f"Error init IDM: {e}")
        # Fallback
        idm = IDM(model_name="mask", output_dim=14).to(device)
        
    if idm_path and os.path.isfile(idm_path):
        try:
            logger.info(f"Loading checkpoint from {idm_path}")
            # Use same loading logic as eval_idm.py
            loaded_dict = torch.load(idm_path, map_location=f'cuda:{device}', weights_only=False)
            if "model_state_dict" in loaded_dict:
                state_dict = loaded_dict["model_state_dict"]
            else:
                state_dict = loaded_dict
            
            # Handle possible key prefixes if wrapped
            # eval_idm.py handles 'module.' or 'backbone.' prefix cleanup if needed, but here simple replacement
            if list(state_dict.keys())[0].startswith("module."): 
                 state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                 
            idm.load_state_dict(state_dict)
            logger.info(f"IDM loaded from {idm_path}")
        except Exception as e:
            logger.error(f"Failed to load state dict: {e}")
            try:
                idm.load_state_dict(state_dict, strict=False)
                logger.info(f"IDM loaded with strict=False")
            except:
                pass
    idm.eval()


    # # 加载 WanTI2V 模型
    # wan_ti2v = wan.WanTI2V(
    #     config=cfg,
    #     checkpoint_dir="/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B",
    #     pt_dir=pt_dir,
    #     device_id=device,
    #     rank=rank,
    #     t5_fsdp=False,
    #     dit_fsdp=False,
    #     use_sp=False,
    #     t5_cpu=False,
    #     convert_model_dtype=True,
    # )


def batch_tensor_to_jpeg_message(tensor):
    tensor = (tensor * 255).to(torch.uint8).cpu()
    jpeg_message_list = []
    for i in range(tensor.shape[0]):
        jpeg_tensor = torchvision.io.encode_jpeg(tensor[i])
        jpeg_message_list.append(b64encode(jpeg_tensor.numpy().tobytes()).decode("utf-8"))
    return jpeg_message_list


def idm_pred(request, imgs):
    global processor
    global idm
    global dino_preprocessor
    
    return_imgs = request.return_imgs
    imgs = imgs.to(next(idm.parameters()).device)
    
    with torch.no_grad():
        if imgs.ndim == 5:
            # Already processed and batched [B, T, C, H, W]
            model_input = imgs
        else:
            # 4D Input: [B, C, H, W] or [T, C, H, W]
            # Use DinoPreprocessor if available?
            # DinoPreprocessor handles PIL or [H,W,C]. Here we have [T, 3, H, W] tensor presumably?
            # get_pred usually processes before calling idm_pred for idm_action. 
            # For vidar mode, we have [T, 3, H, W] from wan.
            
            # Revert to standard processor for now or implement dino logic
            model_input = processor(imgs)
            
            # If using Resnet_plus and input is [T, C, H, W], we usually treat it as [1, T, C, H, W] for single inference?
            # But normally get_pred should prepare 5D for batch inference
            if model_input.ndim == 4 and ("resnet_plus" in str(type(idm.model)) or "Resnet_plus" in str(type(idm.model))): 
                 model_input = model_input.unsqueeze(0)

        actions = idm(model_input, return_mask=return_imgs)
        # Handle tuple output if idm returns (actions, masks)
        if isinstance(actions, tuple):
            actions = actions[0] # Just the actions
            
    actions = json.dumps(actions.cpu().numpy().tolist())
    pred = {"actions": actions}
    if return_imgs:
        pass 
    return pred


def get_pred(request):
    global cfg
    global processor # keep for fallback
    global idm
    global dino_preprocessor

    # Update Preprocessor args based on request
    # This simulates passing args to eval_idm.py
    dino_preprocessor.use_transform = request.use_transform
    dino_preprocessor.init_transforms() # Re-init transform logic (jitter etc)

    # GT Action Bypass or IDM Action
    if request.gt_action_path and request.mode in ["gt_action", "idm_action"]:
        try:
            full_actions_gt = None
            # Load GT actions for both modes (for plotting in idm_action)
            with h5py.File(request.gt_action_path, 'r') as f:
                if 'joint_action' in f and 'vector' in f['joint_action']:
                    full_actions_gt = f['joint_action']['vector'][:]
                elif 'action' in f:
                    full_actions_gt = f['action'][:]
                elif 'qpos' in f:
                    full_actions_gt = f['qpos'][:]
                # If checking multiple keys is needed, add here
            
            if request.mode == "gt_action":
                full_actions = full_actions_gt
                if full_actions is None:
                    logger.error(f"GT path {request.gt_action_path} has no 'joint_action/vector' or 'action'")
                    return None
                    
            elif request.mode == "idm_action":
                 with h5py.File(request.gt_action_path, 'r') as f:
                    if 'observation' in f and 'head_camera' in f['observation'] and 'rgb' in f['observation']['head_camera']:
                        rgb_data = f['observation']['head_camera']['rgb'][:]
                    else:
                        logger.error(f"GT path {request.gt_action_path} has no 'observation/head_camera/rgb'")
                        return None
                 
                 frames_pil = []
                 for i in range(len(rgb_data)):
                     img_bytes = rgb_data[i]
                     try:
                        # Decode bytes to numpy for opencv processing (needed for center_crop_to_aspect)
                        if isinstance(img_bytes, (bytes, np.bytes_)):
                             img_np = cv2.imdecode(np.frombuffer(img_bytes.strip(b'\0'), np.uint8), cv2.IMREAD_COLOR)
                        else:
                             img_np = img_bytes # Assume it's already np array if not bytes

                        if img_np is None:
                            continue

                        # Apply crop and resize if using resnet_plus (following eval_idm.py logic)
                        # We assume crop_and_resize is True for Robotwin/resnet_plus as per test_robotwin.sh
                        if "resnet_plus" in str(type(idm.model)) or "Resnet_plus" in str(type(idm.model)): 
                            target_w, target_h = 832, 480
                            cropped = center_crop_to_aspect(img_np, target_w, target_h)
                            img_np = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Convert to PIL RGB
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_rgb)
                        
                        frames_pil.append(img)
                     except Exception as e:
                         pass 
                 
                 if not frames_pil:
                     return None

                 # Use DinoPreprocessor exactly as eval_idm.py does
                 # It processes PIL images one by one
                 processed_frames = []
                 for img in frames_pil:
                     # process_image returns Tensor [3, 512, 512]
                     processed_frames.append(dino_preprocessor.process_image(img))
                 
                 imgs_processed = torch.stack(processed_frames).to(next(idm.parameters()).device) # [T, 3, 512, 512]
                 
                 # New Logic: Construct batch with [i-2, i, i+2] logic
                 if "Resnet_plus" in str(type(idm.model)) or "resnet_plus" in str(type(idm.model)):
                     T_frames = imgs_processed.shape[0]
                     num_frames = 3
                     idxs_list = [-2, 0, 2] # Relative indices: t-2, t, t+2
                     
                     input_batch = torch.zeros((T_frames, num_frames, 3, 512, 512), device=imgs_processed.device, dtype=imgs_processed.dtype)
                     
                     for i in range(T_frames):
                         # Logic: load [i-2, i, i+2]
                         current_indices = []
                         for offset in idxs_list:
                             idx = i + offset
                             # Clamp index
                             idx = max(0, min(idx, T_frames - 1))
                             current_indices.append(idx)
                        
                         for k, idx in enumerate(current_indices):
                             input_batch[i, k] = imgs_processed[idx]
                     
                     imgs = input_batch # [T, 3, 3, 512, 512]
                 else:
                     # Fallback or other model types (though Resnet_plus seems main one here)
                     imgs = imgs_processed

                 # Run IDM
                 # idm_pred expects imgs and returns dict with actions string
                 pred_result = idm_pred(request, imgs)
                 full_actions = json.loads(pred_result["actions"])
                 full_actions = np.array(full_actions)

                 
                 # --- Plotting and Smoothness Logic ---
                 # Only perform this if we have full GT to compare against and we haven't done it this session?
                 # Or every time? Since this is a stateless request, we do it every request if gt_action_path is new?
                 # Since request might be partial (sliced), but we just computed FULL actions from HDF5.
                 # Let's save plot somewhere.
                 if full_actions_gt is not None:
                     # Align lengths
                     length = min(len(full_actions), len(full_actions_gt))
                     pred_arr = full_actions[:length]
                     gt_arr = full_actions_gt[:length]
                     
                     # Check if we should save plot (maybe use port/seed to differentiate)
                     # Save dir: same as eval_result if possible, or /tmp
                     # We can use os.path.dirname(request.gt_action_path) + "/plots"
                     try:
                         plot_dir = "/data/dex/vidar/vidar-robotwin/eval_result/ar/debug_idm"
                         # Use random component or seed to avoid overwrite if called multiple times?
                         # Or just overwrite for "debug_idm" purpose.
                         # Using seed from request.
                         plot_actions(gt_arr, pred_arr, plot_dir, f"seed{request.seed}_port{request.gt_action_path.split('/')[-1]}") # naming trick
                         
                         score = compute_smoothness(torch.tensor(pred_arr))
                         logger.info(f"Smoothness Score: {score}")
                     except Exception as e:
                         logger.error(f"Plotting failed: {e}")

            
            # Calculate start index (0-based)
            # num_conditional_frames starts at 1 usually. 
            start_idx = max(0, request.num_conditional_frames - 1)
            end_idx = start_idx + request.num_new_frames
            
            sliced_actions = full_actions[start_idx:end_idx]
            
            # For GT replay, avoid padding. Just return available actions.
            # If we run out of actions, we return what we have.
            
            logger.info(f"Using {request.mode} from {request.gt_action_path}, idx {start_idx}:{end_idx}, returned {len(sliced_actions)}")
            return {"actions": json.dumps(sliced_actions.tolist())}

        except Exception as e:
            logger.error(f"Failed to load/process actions in {request.mode}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    elif request.mode == "vidar":
        frame_num = request.num_conditional_frames + request.num_new_frames
        img = request.imgs[-1]
        img = Image.open(io.BytesIO(base64.b64decode(img)))
        img = img.resize(SIZE_CONFIGS["640*736"])
        
        # 生成视频/图像
        imgs = wan_ti2v.generate(
            request.prompt,
            img=img,
            size=SIZE_CONFIGS["640*736"],
            max_area=MAX_AREA_CONFIGS["640*736"],
            frame_num=frame_num,
            shift=cfg.sample_shift,
            sample_solver='unipc',
            sampling_steps=request.num_sampling_step,
            guide_scale=request.guide_scale,
            seed=request.seed,
        )
        imgs = imgs[None].clamp(-1, 1)
        imgs = torch.stack([torchvision.utils.make_grid(u, nrow=8, normalize=True, value_range=(-1, 1)) for u in imgs.unbind(2)], dim=1).permute(1, 0, 2, 3) # [B, C, H, W]
        pred = idm_pred(request, imgs)
        return pred
    
    else:
        logger.error(f"Unknown mode: {request.mode}")
        return None


api = FastAPI()
wan_ti2v = None
ulysses_size = None
cfg = None
idm = None
processor = None
mask_processor = None
init()


@api.post("/")
async def predict(request: Request):
    try:
        # print("Request:", request.prompt, request.num_conditional_frames, request.num_new_frames, request.seed)
        # 简单的鉴权
        if sha256(request.password) == "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307":
            pred = get_pred(request)
            if pred is not None:
                return pred
            else:
                logger.error("get_pred returned None")
                return {"actions": "[]", "error": "get_pred returned None"}
        else:
            return {}
    except Exception as e:
        logger.error(f"Predict error: {e}")
        import traceback
        traceback.print_exc()
        return {"actions": "[]", "error": str(e)}

@api.get("/")
async def test():
    return {"message": "Hello, this is vidar server!"}
