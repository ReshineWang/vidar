import torch
import torch.nn as nn
import os

from .resnet import *
from .unet import *


class IDM(nn.Module):

    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case "unet":
                self.model = Unet(*args, **kwargs)
            case "resnet":
                self.model = ResNet(*args, **kwargs)
            case "dino-resnet":
                self.model = DinoResNet(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
            
        train_mean = torch.tensor([-0.26866713, 0.83559588, 0.69520934, -0.29099351, 0.18849116, -0.01014598, 1.41953145, 0.35073715, 1.05651613, 0.8930193, -0.37493264, -0.18510782, -0.0272574, 1.35274259])
        train_std = torch.tensor([0.25945241, 0.65903812, 0.52147207, 0.42150272, 0.32029947, 0.28452226, 1.78270006, 0.29091741, 0.67675932, 0.58250554, 0.42399049, 0.28697442, 0.31100304, 1.67651926])
        
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class DinoResNet(nn.Module):
    def __init__(self, output_dim=14, dino_version="dinov3_vitl16", *args, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        
        print(f"Loading DinoV3 model from torch.hub: facebookresearch/dinov3 - {dino_version}")
        # Note: This requires network access or cached model to get the structure.
        # We load the structure only (pretrained=False) then load local weights.
        self.backbone = torch.hub.load('facebookresearch/dinov3', dino_version, pretrained=False)
        
        # Load local checkpoint
        ckpt_path = "/data/dex/vidar/vidar_ckpts/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        print(f"Loading local weights from: {ckpt_path}")
        
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            # Handle common DINO wrapper keys
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            
            # Clean up state dictionary keys if necessary (e.g. remove 'backbone.' or 'module.' prefix)
            # This handles cases where the saved model was wrapped in DDP or a larger implementation
            new_state_dict = {}
            for k, v in state_dict.items():
                k_clean = k.replace("module.", "").replace("backbone.", "")
                new_state_dict[k_clean] = v
                # Also keep original just in case
                new_state_dict[k] = v
            
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. {msg}")
        else:
            print(f"WARNING: Local checkpoint not found at {ckpt_path}. Using random initialization!")

        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # Infer embed_dim from the model
        if hasattr(self.backbone, "embed_dim"):
            self.embed_dim = self.backbone.embed_dim
        else:
            # Fallback heuristics
            if "vitl" in dino_version: self.embed_dim = 1024
            elif "vitb" in dino_version: self.embed_dim = 768
            elif "vits" in dino_version: self.embed_dim = 384
            elif "vitg" in dino_version: self.embed_dim = 1536
            else: self.embed_dim = 1024 # Default guess
            
        print(f"Dino backbone loaded. Embed dim: {self.embed_dim}. Backbone is FROZEN.")

        # ResNet head
        # Input to ResNet will be the feature map from Dino
        # shape: (B, Embed_Dim, H_patch, W_patch)
        self.resnet_model = ResNet(output_dim, input_channels=self.embed_dim)
        
        # Print number of parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"DinoResNet initialized. Trainable parameters (ResNet only): {trainable_params}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        # images: [B, 3, H, W]
        # Dino expects normalized images. Our processor does that.
        
        with torch.no_grad():
            # Get features from Dino
            # forward_features returns a dict with 'x_norm_patchtokens' typically for DINOv2/v3
            # x_norm_patchtokens: [B, N, C]
            features_dict = self.backbone.forward_features(images)
            patch_tokens = features_dict['x_norm_patchtokens']
            
            B, N, C = patch_tokens.shape
            # Assuming square image and patch size 14
            # e.g. 518 / 14 = 37. 37*37 = 1369.
            H_grid = W_grid = int(N ** 0.5)
            
            # Reshape to [B, C, H_grid, W_grid]
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_grid, W_grid)
        
        # Pass feature map to ResNet
        outputs = self.resnet_model(feature_map)
        
        if return_mask:
            # This architecture does not predict masks, but IDM interface might ask for it
            return outputs, None
        else:
            return outputs


class Unet(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 3)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, *args, **kwargs):
        outputs = self.resnet_model(torch.tanh(self.mask_model(images)))
        return outputs


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):

        super().__init__()
        self.output_dim = output_dim

        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(14, 3)

        # Print number of parameters
        print(f"output_dim: {output_dim}, parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)

        ## 这里反向传播的时候允许梯度通过mask
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs
