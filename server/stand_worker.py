import os
import torch
import hashlib
from base64 import b64encode
from fastapi import FastAPI
from pydantic import BaseModel
import json
import torchvision
import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from server.idm import IDM
import io
import base64
from PIL import Image
import logging
import h5py
import numpy as np

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
    
    # 直接使用 CUDA_VISIBLE_DEVICES 里的第一个设备 (即 cuda:0)
    device = 0
    
    # 初始化图像处理
    processor = torchvision.transforms.Compose([
        torchvision.transforms.Resize((518, 518)),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mask_processor = torchvision.transforms.Resize((736, 640))
    
    # 加载 IDM 模型
    if idm_path and idm_path.endswith("_out.pt"):
        output_dim = int(idm_path.split("_out.pt")[0].split("_")[-1])
        idm = IDM(model_name="mask", output_dim=output_dim).to(device)
    else:
        idm = IDM(model_name="mask", output_dim=14).to(device)
        
    if idm_path and os.path.isfile(idm_path):
        loaded_dict = torch.load(idm_path, map_location=f'cuda:{device}', weights_only=False)
        idm.load_state_dict(loaded_dict["model_state_dict"])
        logger.info(f"IDM loaded from {idm_path}")
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
        actions, masks = idm(processor(imgs), return_mask=return_imgs)
    actions = json.dumps(actions.cpu().numpy().tolist())
    pred = {"actions": actions}
    if return_imgs:
        pred['imgs'] = batch_tensor_to_jpeg_message(imgs)
        masks = mask_processor(masks)
        pred['masks'] = batch_tensor_to_jpeg_message(torch.where(masks >= 0.5, imgs, 1))
    return pred


def get_pred(request):
    global cfg

    # GT Action Bypass
    if request.gt_action_path:
        try:
            with h5py.File(request.gt_action_path, 'r') as f:
                if 'joint_action' in f and 'vector' in f['joint_action']:
                    full_actions = f['joint_action']['vector'][:]
                elif 'action' in f:
                    full_actions = f['action'][:]
                else:
                    logger.error(f"GT path {request.gt_action_path} has no 'joint_action/vector' or 'action'")
                    return None
            
            # Calculate start index (0-based)
            # num_conditional_frames starts at 1 usually. 
            start_idx = max(0, request.num_conditional_frames - 1)
            end_idx = start_idx + request.num_new_frames
            
            sliced_actions = full_actions[start_idx:end_idx]
            
            # For GT replay, avoid padding. Just return available actions.
            # If we run out of actions, we return what we have.
            
            logger.info(f"Using GT Actions from {request.gt_action_path}, idx {start_idx}:{end_idx}, returned {len(sliced_actions)}")
            return {"actions": json.dumps(sliced_actions.tolist())}

        except Exception as e:
            logger.error(f"Failed to load GT actions: {e}")
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
    print("Request:", request.prompt, request.num_conditional_frames, request.num_new_frames, request.seed)
    # 简单的鉴权
    if sha256(request.password) == "d43e76d9cad30d53805246aa72cc25b04ce2cbe6c7086b53ac6fb5028c48d307":
        pred = get_pred(request)
        if pred is not None:
            return pred
    else:
        return {}

@api.get("/")
async def test():
    return {"message": "Hello, this is vidar server!"}
