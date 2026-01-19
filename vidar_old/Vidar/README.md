# Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation

### 📝[Paper](https://arxiv.org/abs/2507.12898) | 🌍[Project Page](https://embodiedfoundation.github.io/vidar_anypos) | [Pre-trained HunyuanVideo Checkpoint](https://huggingface.co/yaofeng1998/Vidar-Pretrain-10000steps)

Also refer to [here](https://github.com/thu-ml/vidar) for the latest version with the Wan 2.2 model.

## Introduction

Here is the codebase for **Vidar: Embodied Video Diffusion Model for Generalist Bimanual Manipulation**.

Below you will find setup instructions and basic usage guidance for the code within the `vidar` folder.

---

## Environment Setup

Our code has been tested with CUDA 12.4.  
If you encounter errors, please also refer to known issues in [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V).

### 1. Create a Conda Environment
```bash
conda create -n vidar python==3.11.9
```

### 2. Activate the Environment
```bash
conda activate vidar
```

### 3. Install PyTorch and CUDA Dependencies
For CUDA 12.4:
```bash
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
```
*(Optional) Install the full CUDA toolkit:*
```bash
conda install -c nvidia cuda-toolkit=12.4
```

### 4. Install Python Requirements
```bash
python -m pip install -r requirements.txt
```

### 5. Install Flash Attention v2 for Acceleration
Requires CUDA 11.8 or newer:
```bash
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3
```

### 6. Install xDiT for Parallel Inference
We recommend using PyTorch 2.4.0 and flash-attn 2.6.3:
```bash
python -m pip install xfuser==0.4.0
```

#### Troubleshooting: Floating Point Exceptions

If you encounter floating point exceptions (core dump) on certain GPUs, try:
```bash
pip install nvidia-cublas-cu12==12.4.5.8
export LD_LIBRARY_PATH=/opt/conda/lib/python3.8/site-packages/nvidia/cublas/lib/
```
Ensure you have CUDA 12.4, CUBLAS >= 12.4.5.8, and CUDNN >= 9.00 installed.

---

## Video Diffusion Model

### Data Preparation

Prepare your metadata as follows:
```json
{
    "video_path": "{VIDEO_PATH}",
    "raw_caption": {
        "long caption": "{PROMPT}"
    }
}
```

You also need to encode the videos in your dataset before training:
```
vm/hyvae_extract/start.sh
```

For more details, refer to [Hunyuan VAE extract](VIDAR/vm/hyvae_extract/README.md).

### Training

Edit `scripts/vm/train.sh` to match your platform settings, then run:
```bash
scripts/vm/train.sh
```

### Inference

To test your trained model:
```bash
scripts/vm/sample.sh
```
This generates a video based on the first frame and your instruction.

---

## Masked Inverse Dynamic Model

### Data Preparation

- **Training data (default folder):** `assets/train`
- **Testing data (default folder):** `assets/test`
- Files are organized as `task_name/episode_idx.mp4` (a multi-view video) and `task_name/episode_idx_qpos.pt` (a 2D tensor with corresponding actions).

### Training

Edit `scripts/idm/train.sh` as needed, then run:
```bash
scripts/idm/train.sh
```

### Inference

To evaluate your model:
```bash
scripts/idm/eval.sh
```
