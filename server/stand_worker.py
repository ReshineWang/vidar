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

import io
import base64
from PIL import Image
import logging
import h5py
import numpy as np
import cv2

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


def sha256(text):
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def init():
    global wan_ti2v
    global ulysses_size
    global cfg
    global processor
    global mask_processor
    global idm
    
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
        
    model_name = "mask"
    if "resnet_plus" in idm_path or "big_view" in idm_path:
        model_name = "resnet_plus"
    
    # Allow override
    model_name = os.getenv("IDM_MODEL_NAME", model_name)
    
    # 直接使用 CUDA_VISIBLE_DEVICES 里的第一个设备 (即 cuda:0)
    device = 0
    
    # 初始化图像处理
    # Update processor if needed (e.g. for resnet_plus we might want 512)
    if model_name == "resnet_plus":
        # Note: eval_idm.py uses DinoPreprocessor which does resize to 512x512
        # We will manually handle resize in get_pred for idm_action if needed, 
        # or use this processor for standard tensor conversion.
        processor = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("Updated processor to 512x512 for Resnet_plus")
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
            loaded_dict = torch.load(idm_path, map_location=f'cuda:{device}', weights_only=False)
            if "model_state_dict" in loaded_dict:
                state_dict = loaded_dict["model_state_dict"]
            else:
                state_dict = loaded_dict
            
            # Handle possible key prefixes if wrapped
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

    # 加载 WanTI2V 模型
    wan_ti2v = wan.WanTI2V(
        config=cfg,
        checkpoint_dir="/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B",
        pt_dir=pt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        convert_model_dtype=True,
    )


def batch_tensor_to_jpeg_message(tensor):
    tensor = (tensor * 255).to(torch.uint8).cpu()
    jpeg_message_list = []
    for i in range(tensor.shape[0]):
        jpeg_tensor = torchvision.io.encode_jpeg(tensor[i])
        jpeg_message_list.append(b64encode(jpeg_tensor.numpy().tobytes()).decode("utf-8"))
    return jpeg_message_list


def idm_pred(request, imgs):
    global processor
    global mask_processor
    global idm
    return_imgs = request.return_imgs
    imgs = imgs.to(next(idm.parameters()).device)
    
    with torch.no_grad():
        if imgs.ndim == 5:
            # Already processed and batched [B, T, C, H, W]
            model_input = imgs
        else:
            model_input = processor(imgs)
            # If using Resnet_plus and input is [T, C, H, W], we usually treat it as [1, T, C, H, W] for single inference?
            # But normally get_pred should prepare 5D for batch inference
            if model_input.ndim == 4 and "resnet_plus" in str(type(idm.model)): # basic check
                 model_input = model_input.unsqueeze(0)

        actions = idm(model_input, return_mask=return_imgs)
        # Handle tuple output if idm returns (actions, masks)
        if isinstance(actions, tuple):
            actions = actions[0] # Just the actions
            
    actions = json.dumps(actions.cpu().numpy().tolist())
    pred = {"actions": actions}
    if return_imgs:
        # Note: this might fail for 5D imgs, but return_imgs=False for idm_action usually
        pass 
    return pred


def get_pred(request):
    global cfg

    # GT Action Bypass or IDM Action
    if request.gt_action_path and request.mode in ["gt_action", "idm_action"]:
        try:
            if request.mode == "gt_action":
                with h5py.File(request.gt_action_path, 'r') as f:
                    if 'joint_action' in f and 'vector' in f['joint_action']:
                        full_actions = f['joint_action']['vector'][:]
                    elif 'action' in f:
                        full_actions = f['action'][:]
                    else:
                        logger.error(f"GT path {request.gt_action_path} has no 'joint_action/vector' or 'action'")
                        return None
            elif request.mode == "idm_action":
                 with h5py.File(request.gt_action_path, 'r') as f:
                    if 'observation' in f and 'head_camera' in f['observation'] and 'rgb' in f['observation']['head_camera']:
                        rgb_data = f['observation']['head_camera']['rgb'][:]
                    else:
                        logger.error(f"GT path {request.gt_action_path} has no 'observation/head_camera/rgb'")
                        return None
                 
                 frames = []
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
                        if "resnet_plus" in str(type(idm.model)):
                            target_w, target_h = 832, 480
                            cropped = center_crop_to_aspect(img_np, target_w, target_h)
                            img_np = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                        
                        # Convert to PIL RGB
                        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(img_rgb)
                        
                        frames.append(torchvision.transforms.functional.to_tensor(img))
                     except Exception as e:
                         # logger.warning(f"Frame decode error: {e}")
                         pass 
                 
                 if not frames:
                     return None

                 imgs = torch.stack(frames).to(next(idm.parameters()).device) # [T, 3, H, W]
                 
                 # Special handling for Resnet_plus sequence input
                 if "Resnet_plus" in str(type(idm.model)):
                     imgs_processed = processor(imgs) # [T, 3, 512, 512]
                     num_frames = 3
                     T_frames = imgs_processed.shape[0]
                     input_batch = torch.zeros((T_frames, num_frames, 3, 512, 512), device=imgs_processed.device, dtype=imgs_processed.dtype)
                     
                     for i in range(T_frames):
                         # Logic: for frame t, we want [t-2, t-1, t]. If t < 2, pad with frame 0.
                         # My previous logic: i - (num_frames - 1 - j)
                         # j=0 -> i-2, j=1 -> i-1, j=2 -> i
                         for j in range(num_frames):
                             idx = i - (num_frames - 1 - j)
                             if idx < 0:
                                 idx = 0 # Repeat first frame for padding
                             input_batch[i, j] = imgs_processed[idx]
                     imgs = input_batch # [T, 3, 3, 512, 512]
                 
                 # Run IDM
                 # idm_pred expects imgs and returns dict with actions string
                 pred_result = idm_pred(request, imgs)
                 full_actions = json.loads(pred_result["actions"])
                 full_actions = np.array(full_actions)
            
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
